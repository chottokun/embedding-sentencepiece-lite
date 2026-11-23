"""lite セットの検証スクリプト

生成された lite セット (model.onnx + .spm.fb + loader + lite_config.json) について、
1. ローダーのトークン ID 列が SentencePiece 生 ID + 実測済み BOS/EOS と一致すること
2. 埋め込みが参照実装 (sentence-transformers, FP32) とコサイン類似度で一致すること
3. HF_HUB_OFFLINE=1 の別プロセスでも Hub を呼ばずに埋め込みが生成でき、通常時と一致すること
を検証する。PyTorch / Transformers はこの検証側でのみ使用する。

使用例:
  uv run python scripts/verify_lite_set.py \
    --model-dir ~/lite_factory_outputs/multilingual_e5_base \
    --base-model intfloat/multilingual-e5-base
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SAMPLES = [
    "The capital of Japan is Tokyo.",  # 英語
    "日本の首都は東京です。",  # 日本語
    "Embedding models convert text into vectors. テキストをベクトルに変換します。",  # 混在
]
THRESHOLD = 0.999


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


def main() -> int:
    parser = argparse.ArgumentParser(description="lite セット検証")
    parser.add_argument("--model-dir", required=True, help="lite セットのディレクトリ")
    parser.add_argument("--base-model", required=True, help="参照実装の HF モデル ID")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    with open(os.path.join(model_dir, "lite_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"検証対象: {model_dir} (base: {args.base_model})")
    print(f"lite_config: bos={cfg['bos_id']} eos={cfg['eos_id']} pad={cfg['pad_id']} "
          f"add_bos={cfg['add_bos']} add_eos={cfg['add_eos']} dim={cfg['embedding_dim']}")

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

    # --- 2. 埋め込み検証 (参照: sentence-transformers FP32) ---
    import torch
    from sentence_transformers import SentenceTransformer

    try:
        ref = SentenceTransformer(args.base_model, device="cpu",
                                  model_kwargs={"dtype": torch.float32})
    except TypeError:
        ref = SentenceTransformer(args.base_model, device="cpu")
    try:
        ref_dtype = next(ref[0].auto_model.parameters()).dtype
    except Exception:
        ref_dtype = None
    print(f"参照実装の dtype: {ref_dtype}")

    lite_raw_emb = lite.encode(SAMPLES)
    cases = [
        ("query", [cfg["query_prefix"] + s for s in SAMPLES], lite.encode_query(SAMPLES)),
        ("passage", [cfg["passage_prefix"] + s for s in SAMPLES], lite.encode_passages(SAMPLES)),
        ("raw", SAMPLES, lite_raw_emb),
    ]
    all_ok = tok_ok
    for name, ref_texts, lite_emb in cases:
        ref_emb = ref.encode(ref_texts, normalize_embeddings=True)
        cos = cos_rows(np.asarray(ref_emb, dtype=np.float32), np.asarray(lite_emb, dtype=np.float32))
        for text, c in zip(SAMPLES, cos):
            ok = bool(c >= THRESHOLD)
            all_ok = all_ok and ok
            print(f"  [{name}] cos={c:.6f} {'OK ' if ok else 'NG '} {text[:24]}...")
    # --- 3. オフラインスモーク (HF_HUB_OFFLINE=1・Hub 呼び出し遮断で通常時と一致するか) ---
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

    print(f"総合判定: {'PASS' if all_ok else 'FAIL'} (閾値 cos >= {THRESHOLD})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
