"""lite セットの検証スクリプト

生成された lite セット (model.onnx + model_fp16.onnx + .spm.fb + loader + lite_config.json) について、
1. ローダーのトークン ID 列が SentencePiece 生 ID + 実測済み BOS/EOS と一致すること
2. 埋め込み (FP32) が参照実装 (sentence-transformers または AutoModel, FP32) とコサイン類似度で一致すること
3. FP16 経路の精度検証 (モデル別しきい値) およびモデル別 GPU 精度ポリシー (allow_fp16 / gpu_precision) が正しく機能すること
4. HF_HUB_OFFLINE=1 の別プロセスでも Hub を呼ばずに埋め込みが生成でき、通常時と一致すること
を検証する。PyTorch / Transformers はこの検証側でのみ使用する。

使用例:
  uv run python scripts/verify_lite_set.py \
    --model-dir ./dist_assets/multilingual_e5_base \
    --base-model intfloat/multilingual-e5-base
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

SAMPLES = [
    "The capital of Japan is Tokyo.",  # 英語
    "日本の首都は東京です。",  # 日本語
    "Embedding models convert text into vectors. テキストをベクトルに変換します。",  # 混在
]
THRESHOLD = 0.999
DEFAULT_FP16_THRESHOLD = 0.995


def find_loader_path(model_dir: str) -> str:
    for name in ("lite_embed.py", "ruri_v3_lite.py"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"ローダーが見つかりません: {model_dir}")


def load_loader_module(model_dir: str):
    path = find_loader_path(model_dir)
    spec = importlib.util.spec_from_file_location("lite_loader", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_OFFLINE_CHILD = """
import importlib.util, json, sys
import numpy as np

model_dir, loader_path, samples_json, out_path, mode = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("lite_loader_offline", loader_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _blocked(*args, **kwargs):
    raise AssertionError("hf_hub_download was called during offline smoke test")


mod.hf_hub_download = _blocked
samples = json.loads(samples_json)
if mode == "explicit":
    lite = mod.LiteEmbedder(model_dir=model_dir, device="cpu")
else:
    lite = mod.LiteEmbedder(device="cpu")
np.save(out_path, np.asarray(lite.encode(samples), dtype=np.float32))
"""


def run_offline_smoke(model_dir: str, loader_path: str, mode: str) -> np.ndarray:
    """HF_HUB_OFFLINE=1 の別プロセスで lite 埋め込みを生成して返す。

    子プロセスでは loader の hf_hub_download を例外送出に差し替えるため、
    ローカルにファイルが揃っていれば Hub を一切呼ばずに完走できることの
    検証にもなる。mode="local" は model_dir を渡さず、ローダー自身の
    ディレクトリからの解決だけに頼る。
    """
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory() as td:
        child = os.path.join(td, "offline_child.py")
        with open(child, "w", encoding="utf-8") as f:
            f.write(_OFFLINE_CHILD)
        out = os.path.join(td, "emb.npy")
        r = subprocess.run(
            [sys.executable, child, model_dir, loader_path,
             json.dumps(SAMPLES), out, mode],
            env=env, capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            raise RuntimeError(f"オフラインスモーク実行失敗 (mode={mode})")
        return np.load(out)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a * b, axis=1)


class _AutoModelMeanPoolingRef:
    """SentenceTransformer がバージョン非互換や画像プロセッサ不足で読み込めない場合の
    AutoTokenizer + AutoModel (または language_model) + Mean Pooling 参照実装フォールバック"""

    def __init__(self, base_model: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        try:
            full_model = AutoModel.from_pretrained(base_model, dtype=torch.float32)
        except TypeError:
            full_model = AutoModel.from_pretrained(base_model, torch_dtype=torch.float32)
        full_model.eval()
        self.backbone = full_model.language_model if hasattr(full_model, "language_model") else full_model
        self.dtype = next(self.backbone.parameters()).dtype

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


def load_reference_model(base_model: str):
    """SentenceTransformer を優先し、非互換時は AutoModel + Mean Pooling にフォールバックする"""
    import torch

    try:
        from sentence_transformers import SentenceTransformer

        try:
            ref = SentenceTransformer(base_model, device="cpu", model_kwargs={"dtype": torch.float32})
        except TypeError:
            ref = SentenceTransformer(base_model, device="cpu")
        try:
            ref_dtype = next(ref[0].auto_model.parameters()).dtype
        except Exception:
            ref_dtype = None
        print(f"参照実装: SentenceTransformer (dtype={ref_dtype})")
        return ref
    except Exception as e:
        print(f"  [info] SentenceTransformer でのロード失敗 ({type(e).__name__}: {e}) -> AutoModel + MeanPooling 参照へフォールバック")
        ref = _AutoModelMeanPoolingRef(base_model)
        print(f"参照実装: AutoModel + MeanPooling (dtype={ref.dtype})")
        return ref


def main() -> int:
    parser = argparse.ArgumentParser(description="lite セット検証")
    parser.add_argument("--model-dir", required=True, help="lite セットのディレクトリ")
    parser.add_argument("--base-model", required=True, help="参照実装の HF モデル ID")
    parser.add_argument("--fp16-threshold", type=float, default=None, help="FP16 経路の cos 合格閾値 (未指定時は lite_config または 0.995)")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    cfg_file = os.path.join(model_dir, "lite_config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        # ruri-v3 従来ディレクトリ (lite_config.json 不在) 用の既定値
        cfg = {
            "bos_id": 1,
            "eos_id": 2,
            "pad_id": 3,
            "add_bos": True,
            "add_eos": True,
            "query_prefix": "検索クエリ: ",
            "passage_prefix": "文章: ",
            "embedding_dim": None,
            "gpu_precision": "fp16",
            "allow_fp16": True,
            "fp16_cos_threshold": DEFAULT_FP16_THRESHOLD,
        }
    print(f"検証対象: {model_dir} (base: {args.base_model})")
    print(
        f"lite_config: bos={cfg['bos_id']} eos={cfg['eos_id']} pad={cfg['pad_id']} "
        f"add_bos={cfg['add_bos']} add_eos={cfg['add_eos']} dim={cfg.get('embedding_dim')} "
        f"gpu_precision={cfg.get('gpu_precision', 'fp16')} allow_fp16={cfg.get('allow_fp16', True)}"
    )

    # --- 1. トークナイザ検証 (SentencePiece 生 ID + BOS/EOS) ---
    import sentencepiece as sp_lib

    sp_path = None
    for cand in ("tokenizer.model", "sentencepiece.bpe.model"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            sp_path = p
            break
    sp = sp_lib.SentencePieceProcessor()
    sp.Load(sp_path)

    mod = load_loader_module(model_dir)
    lite = mod.LiteEmbedder(model_dir=model_dir, device="cpu")

    tok_ok = True
    for text in SAMPLES:
        expected = sp.Encode(text)
        if cfg["add_bos"]:
            expected = [cfg["bos_id"]] + expected
        if cfg["add_eos"]:
            expected = expected + [cfg["eos_id"]]
        got = lite.tokenizer.encode(text)
        got = ([cfg["bos_id"]] if cfg["add_bos"] else []) + got + ([cfg["eos_id"]] if cfg["add_eos"] else [])
        match = got == expected
        tok_ok = tok_ok and match
        print(f"  [tokenizer] {'OK ' if match else 'NG '} {text[:24]}... ids={len(expected)}")
    print(f"トークナイザ検証: {'PASS' if tok_ok else 'FAIL'}")

    # --- 2. 埋め込み検証 (FP32 vs 参照 FP32) ---
    ref = load_reference_model(args.base_model)

    lite_raw_emb = lite.encode(SAMPLES)
    cases = [
        ("query", [cfg["query_prefix"] + s for s in SAMPLES], lite.encode_query(SAMPLES)),
        ("passage", [cfg["passage_prefix"] + s for s in SAMPLES], lite.encode_passages(SAMPLES)),
        ("raw", SAMPLES, lite_raw_emb),
    ]
    ref_cache = {}
    all_ok = tok_ok
    for name, ref_texts, lite_emb in cases:
        ref_emb = np.asarray(ref.encode(ref_texts, normalize_embeddings=True), dtype=np.float32)
        ref_cache[name] = ref_emb
        cos = cos_rows(ref_emb, np.asarray(lite_emb, dtype=np.float32))
        for text, c in zip(SAMPLES, cos):
            ok = bool(c >= THRESHOLD)
            all_ok = all_ok and ok
            print(f"  [{name}:fp32] cos={c:.6f} {'OK ' if ok else 'NG '} {text[:24]}...")

    # --- 3. FP16 経路および GPU 精度ポリシー検証 ---
    allow_fp16 = bool(cfg.get("allow_fp16", True)) and (cfg.get("gpu_precision", "fp16") == "fp16")
    fp16_threshold = (
        args.fp16_threshold
        if args.fp16_threshold is not None
        else (cfg.get("fp16_cos_threshold") or DEFAULT_FP16_THRESHOLD)
    )
    print(f"FP16 経路・GPU 精度ポリシー検証 (allow_fp16={allow_fp16}, 閾値 cos >= {fp16_threshold}):")

    # 3-a. CC >= 7.0 (例: Tesla T4 CC 7.5) 環境をシミュレートして既定のモデル選択を検査
    class _DummyOutput:
        shape = [1, 1, cfg.get("embedding_dim") or 256]

    class _DummySession:
        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def get_outputs(self):
            return [_DummyOutput()]

    with patch.object(mod, "get_cuda_compute_capability", return_value=7.5), patch.object(
        mod.ort, "InferenceSession", return_value=_DummySession()
    ):
        policy_probe = mod.LiteEmbedder(model_dir=model_dir, device="cuda")
    expected_filename = (cfg.get("onnx_fp16") or "model_fp16.onnx") if allow_fp16 else (cfg.get("onnx") or "model.onnx")
    selected_filename = getattr(policy_probe, "model_filename", None)
    policy_ok = selected_filename == expected_filename
    all_ok = all_ok and policy_ok
    print(
        f"  [policy:CC=7.5] selected={selected_filename} expected={expected_filename} "
        f"{'OK ' if policy_ok else 'NG '}"
    )

    # 3-b. model_fp16.onnx 経路のコサイン類似度チェック (CC >= 7.0 の GPU があれば CUDA、なければ CPU)
    fp16_onnx_path = os.path.join(model_dir, cfg.get("onnx_fp16") or "model_fp16.onnx")
    if os.path.exists(fp16_onnx_path):
        has_cc7_cuda = (
            "CUDAExecutionProvider" in mod.ort.get_available_providers()
            and mod.get_cuda_compute_capability() >= 7.0
        )
        fp16_device = "cuda" if has_cc7_cuda else "cpu"
        lite_fp16 = mod.LiteEmbedder(model_dir=model_dir, device=fp16_device, force_fp16=True)
        fp16_cases = [
            ("query", lite_fp16.encode_query(SAMPLES)),
            ("passage", lite_fp16.encode_passages(SAMPLES)),
            ("raw", lite_fp16.encode(SAMPLES)),
        ]
        for name, fp16_emb in fp16_cases:
            cos = cos_rows(ref_cache[name], np.asarray(fp16_emb, dtype=np.float32))
            min_c = float(cos.min())
            if allow_fp16:
                ok = bool(min_c >= fp16_threshold)
                all_ok = all_ok and ok
                print(f"  [{name}:fp16] cos min={min_c:.6f} {'OK ' if ok else 'NG '} (閾値 >= {fp16_threshold})")
            else:
                print(
                    f"  [{name}:fp16] cos min={min_c:.6f} "
                    f"(参考値: gpu_precision=fp32 により既定では FP32 が強制されるため合否対象外)"
                )
    elif allow_fp16:
        print(f"  [fp16] NG: allow_fp16=True ですが {fp16_onnx_path} が存在しません")
        all_ok = False

    # --- 4. オフラインスモーク (HF_HUB_OFFLINE=1・Hub 呼び出し遮断で通常時と一致するか) ---
    print("オフラインスモーク (HF_HUB_OFFLINE=1, Hub 呼び出し遮断):")
    loader_path = find_loader_path(model_dir)
    for mode in ("explicit", "local"):
        try:
            off_emb = run_offline_smoke(model_dir, loader_path, mode)
            cos = cos_rows(np.asarray(lite_raw_emb, dtype=np.float32), off_emb)
            ok = bool(np.all(cos >= THRESHOLD))
            all_ok = all_ok and ok
            print(f"  [offline:{mode}] cos min={cos.min():.6f} {'OK ' if ok else 'NG '}")
        except Exception as e:
            all_ok = False
            print(f"  [offline:{mode}] 実行失敗: {e}")

    print(f"総合判定: {'PASS' if all_ok else 'FAIL'} (FP32 閾値 >= {THRESHOLD}, FP16 閾値 >= {fp16_threshold})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
