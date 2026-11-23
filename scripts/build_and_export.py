"""資産生成スクリプト (マルチモデル対応版)
ruri-v3 シリーズ (30m, 70m, 130m, 310m) に加え、SentencePiece 系の埋め込みモデル
(multilingual-e5-base, embeddinggemma-2) の lite 資産を生成する。

モデルごとの差異 (HF repo・トークナイザファイル・接頭辞・プーリング方式・
ライセンス・ONNX 書き出し対象) は MODEL_CONFIGS の設定表に集約する。
特殊トークン ID と BOS/EOS 付与の有無はハードコードせず、ビルド時に
AutoTokenizer と SentencePiece モデル自体から実測して lite_config.json に記録する。
書き出された ONNX は SentencePiece ネイティブの ID 空間で動作する
(HF 側の ID 空間と異なるモデルでは、埋め込み行列の行を置換して吸収する)。

使用例:
  uv run python scripts/build_and_export.py --model 70m
  uv run python scripts/build_and_export.py --model e5-base
  uv run python scripts/build_and_export.py --model gemma2 --out-root ~/lite_factory_outputs
  uv run python scripts/build_and_export.py --all
"""
import json
import os
import sys
import shutil
import argparse
import subprocess
import onnx
import torch
import sentencepiece as sp_lib
from onnxconverter_common import float16
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from onnx_utils import cleanup_external_data, sanitize_fp16_cast_nodes, verify_inline_onnx


def _ruri_cfg(key: str) -> dict:
    return {
        "repo_id": f"cl-nagoya/ruri-v3-{key}",
        "family": "ruri-v3",
        "display_name": f"ruri-v3-{key}",
        "spm_filename": "tokenizer.model",
        "fb_filename": f"ruri_v3_{key}.spm.fb",
        "out_dirname": f"ruri_v3_{key}",
        # ruri-v3 公式の推奨接頭辞
        "query_prefix": "検索クエリ: ",
        "passage_prefix": "文章: ",
        "pooling": "mean",
        "license": "apache-2.0",
        "export_target": "auto",
    }


MODEL_CONFIGS = {
    "30m": _ruri_cfg("30m"),
    "70m": _ruri_cfg("70m"),
    "130m": _ruri_cfg("130m"),
    "310m": _ruri_cfg("310m"),
    "e5-base": {
        "repo_id": "intfloat/multilingual-e5-base",
        "family": "multilingual-e5",
        "display_name": "multilingual-e5-base",
        "spm_filename": "sentencepiece.bpe.model",
        "fb_filename": "multilingual_e5_base.spm.fb",
        "out_dirname": "multilingual_e5_base",
        # E5 公式の推奨接頭辞 (モデルカード記載)
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "pooling": "mean",
        "license": "mit",
        "export_target": "auto",
    },
    "gemma2": {
        "repo_id": "google/embeddinggemma-2",
        "family": "embeddinggemma",
        "display_name": "embeddinggemma-2",
        "spm_filename": "tokenizer.model",
        "fb_filename": "embeddinggemma_2.spm.fb",
        "out_dirname": "embeddinggemma_2",
        # config_sentence_transformers.json の prompts から検索用途を採用:
        #   "query" -> "task: search result | query: "
        #   "document" -> "title: none | text: " (タイトル無し文書)
        "query_prefix": "task: search result | query: ",
        "passage_prefix": "title: none | text: ",
        "pooling": "mean",
        "license": "apache-2.0",
        # マルチモーダル全体ではなくテキスト塔 (language_model) のみを書き出す。
        # テキスト入力に対する出力が全体モデルと完全一致することは検証済み。
        "export_target": "language_model",
    },
}

# 後方互換 (従来の SUPPORTED_MODELS 参照用)
SUPPORTED_MODELS = {k: v["repo_id"] for k, v in MODEL_CONFIGS.items()}

BASE_DIST_DIR = os.path.abspath("./dist_assets")
BUILD_DIR = os.path.abspath("./build_spm")
GLOBAL_WHEELS_DIR = os.path.join(BASE_DIST_DIR, "wheels")

os.makedirs(BASE_DIST_DIR, exist_ok=True)
os.makedirs(GLOBAL_WHEELS_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

def ensure_sentencepiece_tools_and_wheel():
    """sentencepiece_lite の C++ ツール (spm_to_fb) および Wheel をビルド (未作成時のみ)"""
    spm_repo_dir = os.path.join(BUILD_DIR, "sentencepiece")
    if not os.path.exists(spm_repo_dir):
        print("Cloning google/sentencepiece...")
        subprocess.run([
            "git", "clone", "--depth=1", "--recurse-submodules", "--shallow-submodules",
            "https://github.com/google/sentencepiece.git", spm_repo_dir
        ], check=True)

    spm_build_dir = os.path.join(spm_repo_dir, "build")
    spm_to_fb_bin = os.path.join(spm_build_dir, "lite", "spm_to_fb")
    if not os.path.exists(spm_to_fb_bin):
        print("Building sentencepiece and spm_to_fb with cmake...")
        subprocess.run([
            "cmake", "-B", spm_build_dir, "-S", spm_repo_dir,
            "-DSPM_ENABLE_SHARED=OFF", "-DSPM_BUILD_TEST=OFF", "-DCMAKE_BUILD_TYPE=Release"
        ], check=True)
        subprocess.run([
            "cmake", "--build", spm_build_dir, "--config", "Release", "-j", str(os.cpu_count() or 4)
        ], check=True)

    whl_files = [os.path.join(GLOBAL_WHEELS_DIR, f) for f in os.listdir(GLOBAL_WHEELS_DIR) if f.endswith(".whl")]
    if not whl_files:
        print("Building sentencepiece_lite Python wheel...")
        pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages/sentencepiece_lite"))
        if not os.path.exists(pkg_dir):
            pkg_dir = os.path.join(BUILD_DIR, "sentencepiece_lite_pkg")
        subprocess.run([
            sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", GLOBAL_WHEELS_DIR, pkg_dir
        ], check=True)
        whl_files = [os.path.join(GLOBAL_WHEELS_DIR, f) for f in os.listdir(GLOBAL_WHEELS_DIR) if f.endswith(".whl")]

    print(f"sentencepiece_lite wheel ready: {whl_files[0]}")
    print(f"spm_to_fb binary ready: {spm_to_fb_bin}")
    return spm_to_fb_bin, whl_files[0]


def _hf_id_of_piece(tokenizer, piece: str) -> int:
    return tokenizer.convert_tokens_to_ids([piece])[0]


def probe_special_tokens(tokenizer, sp, probe_text: str = "検証用テキストです。"):
    """AutoTokenizer の出力と SentencePiece の生 ID 列を突き合わせ、
    BOS/EOS 付与パターンと ID 空間 (同一 or HF 側オフセット) を実測する。

    Returns:
        dict: add_bos, add_eos, id_space ("sentencepiece" | "hf"),
              auto_ids, raw_ids
    """
    raw_ids = sp.Encode(probe_text)
    auto_ids = tokenizer(probe_text)["input_ids"]
    mapped_ids = [_hf_id_of_piece(tokenizer, sp.id_to_piece(i)) for i in raw_ids]

    def _structure(candidate_ids):
        add_bos = bool(
            tokenizer.bos_token_id is not None
            and auto_ids[:1] == [tokenizer.bos_token_id]
            and auto_ids[1:1 + len(candidate_ids)] == candidate_ids
        )
        rest = auto_ids[1:] if add_bos else auto_ids
        add_eos = bool(
            tokenizer.eos_token_id is not None
            and rest[-1:] == [tokenizer.eos_token_id]
            and rest[:len(candidate_ids)] == candidate_ids
        )
        core = rest[:len(candidate_ids)] if add_eos else rest
        return add_bos, add_eos, core == candidate_ids

    add_bos, add_eos, ok = _structure(raw_ids)
    if ok:
        return {"add_bos": add_bos, "add_eos": add_eos, "id_space": "sentencepiece",
                "auto_ids": auto_ids, "raw_ids": raw_ids}

    add_bos, add_eos, ok = _structure(mapped_ids)
    if ok:
        return {"add_bos": add_bos, "add_eos": add_eos, "id_space": "hf",
                "auto_ids": auto_ids, "raw_ids": raw_ids}

    raise RuntimeError(
        "トークナイザの ID 対応が想定外です。"
        f" auto={auto_ids} raw(sp)={raw_ids} mapped={mapped_ids}"
    )


def build_id_permutation(tokenizer, sp):
    """SentencePiece の piece id -> HF の token id への置換表を構築する。
    戻り値の perm[sp_id] が、その sp_id に対応する HF 側の埋め込み行番号。"""
    pieces = [sp.id_to_piece(i) for i in range(sp.GetPieceSize())]
    perm = tokenizer.convert_tokens_to_ids(pieces)
    if len(perm) != len(pieces):
        raise RuntimeError("convert_tokens_to_ids の戻り値長が語彙数と一致しません")
    unk_hf = tokenizer.unk_token_id
    suspicious = sum(
        1 for piece, hf_id in zip(pieces, perm)
        if hf_id == unk_hf and piece != tokenizer.unk_token
    )
    if suspicious:
        print(f"  [warn] HF 語彙に見つからず unk 扱いになった piece が {suspicious} 件あります")
    return perm


def apply_id_permutation(model, perm, pad_hf_id: int) -> int:
    """埋め込み行列の行を SentencePiece ネイティブ ID 空間に置換し、
    末尾に HF の pad 行を追加する。戻り値は新しい pad の ID (= 行数 - 1)。"""
    emb = model.get_input_embeddings()
    W = emb.weight.data
    index = torch.tensor(perm, dtype=torch.long)
    new_W = torch.cat([W[index], W[pad_hf_id:pad_hf_id + 1]], dim=0).clone()
    emb.weight = torch.nn.Parameter(new_W, requires_grad=False)
    if hasattr(emb, "num_embeddings"):
        emb.num_embeddings = new_W.shape[0]
    return new_W.shape[0] - 1


class _BackboneWrapper(torch.nn.Module):
    """(input_ids, attention_mask) -> last_hidden_state のみを返す薄いラッパー"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]


class _PermutedBackboneWrapper(torch.nn.Module):
    """ID 空間置換済みモデル用ラッパー

    XLM-R 系は position_ids を input_ids と padding_idx の比較から生成するため、
    埋め込み行列だけを SentencePiece ID 空間に置換すると、SP 空間で padding_idx と
    同じ値を持つトークン (例: e5 では <s>=1) が padding と誤認され、全トークンの
    position がずれこ。そこで position_ids を「新しい pad id でマスクし、
    オフセットは HF 側 padding_idx」という元の意味論で明示的に計算して渡す。
    """

    def __init__(self, backbone, sp_pad_id: int, position_offset: int):
        super().__init__()
        self.backbone = backbone
        self.sp_pad_id = sp_pad_id
        self.position_offset = position_offset

    def forward(self, input_ids, attention_mask):
        mask = input_ids.ne(self.sp_pad_id).long()
        position_ids = torch.cumsum(mask, dim=1) * mask + self.position_offset
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]


def _load_hf_model(repo_id: str):
    try:
        return AutoModel.from_pretrained(repo_id, dtype=torch.float32)
    except TypeError:
        return AutoModel.from_pretrained(repo_id, torch_dtype=torch.float32)


def _fetch_max_seq_length(repo_id: str):
    try:
        path = hf_hub_download(repo_id=repo_id, filename="sentence_bert_config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("max_seq_length")
    except Exception:
        return None


def export_model(model_key: str, spm_to_fb_bin: str, wheel_path: str, out_root: str = None):
    cfg = MODEL_CONFIGS[model_key]
    repo_id = cfg["repo_id"]
    print(f"\n==================================================")
    print(f"  処理開始: {repo_id} ({model_key})")
    print(f"==================================================")

    base_out_dir = os.path.abspath(out_root) if out_root else BASE_DIST_DIR
    out_dir = os.path.join(base_out_dir, cfg["out_dirname"])
    os.makedirs(out_dir, exist_ok=True)
    wheels_sub_dir = os.path.join(out_dir, "wheels")
    os.makedirs(wheels_sub_dir, exist_ok=True)
    shutil.copyfile(wheel_path, os.path.join(wheels_sub_dir, os.path.basename(wheel_path)))

    # 1. トークナイザ辞書の取得と FlatBuffers (.spm.fb) 変換
    print(f"[{model_key}] 1/5: Tokenizer 辞書の取得と FlatBuffers 変換")
    dst_sp = os.path.join(out_dir, cfg["spm_filename"])
    if not os.path.exists(dst_sp):
        sp_path = hf_hub_download(repo_id=repo_id, filename=cfg["spm_filename"])
        shutil.copyfile(sp_path, dst_sp)
        os.chmod(dst_sp, 0o644)

    fb_path = os.path.join(out_dir, cfg["fb_filename"])
    if not os.path.exists(fb_path):
        print(f"Converting {dst_sp} -> {fb_path}...")
        subprocess.run([
            spm_to_fb_bin,
            f"--model={dst_sp}",
            f"--output={fb_path}"
        ], check=True)
    print(f"FlatBuffers: {fb_path} ({os.path.getsize(fb_path):,} bytes)")

    # 2. 特殊トークンの実測 (ハードコード禁止: モデル側定義から引く)
    print(f"[{model_key}] 2/5: 特殊トークンと ID 空間の実測")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(out_dir)
    if model_key == "30m" and not out_root:
        tokenizer.save_pretrained(BASE_DIST_DIR)
    sp = sp_lib.SentencePieceProcessor()
    sp.Load(dst_sp)
    probe = probe_special_tokens(tokenizer, sp)
    print(f"  add_bos={probe['add_bos']} add_eos={probe['add_eos']} id_space={probe['id_space']}")
    print(f"  AutoTokenizer: bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} "
          f"pad={tokenizer.pad_token_id} unk={tokenizer.unk_token_id}")
    print(f"  SentencePiece: bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()} unk={sp.unk_id()}")

    # 3. FP32 ONNX エクスポート
    print(f"[{model_key}] 3/5: FP32 ONNX エクスポート")
    model = _load_hf_model(repo_id)
    model.eval()

    permuted = False
    new_pad_id = None
    if probe["id_space"] == "hf":
        if tokenizer.pad_token_id is None:
            raise RuntimeError("ID 空間が HF 側なのに pad_token_id がありません")
        perm = build_id_permutation(tokenizer, sp)
        new_pad_id = apply_id_permutation(model, perm, tokenizer.pad_token_id)
        permuted = True
        print(f"  埋め込み行列を SentencePiece ID 空間に置換しました (pad_id={new_pad_id})")

    backbone = model
    if cfg["export_target"] == "language_model":
        backbone = model.language_model
        # テキスト塔のみの出力が全体モデルと一致することを確認してから書き出す
        enc = tokenizer(["検証用テキストです。"], return_tensors="pt")
        with torch.no_grad():
            full = model(**enc).last_hidden_state
            part = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            part = part.last_hidden_state if hasattr(part, "last_hidden_state") else part[0]
        max_diff = (full - part).abs().max().item()
        print(f"  テキスト塔の等価性チェック: max abs diff = {max_diff}")
        if max_diff > 1e-5:
            raise RuntimeError("テキスト塔の出力が全体モデルと一致しません")

    if permuted:
        position_offset = int(model.config.pad_token_id)
        wrapper = _PermutedBackboneWrapper(backbone, new_pad_id, position_offset)
        print(f"  position_ids 明示計算ラッパーを使用 (pad={new_pad_id}, offset={position_offset})")
    else:
        wrapper = _BackboneWrapper(backbone)

    # ダミー入力は lite 側の実運用形式 (SentencePiece 生 ID + 実測した BOS/EOS) で作る
    dummy_raw = sp.Encode("テストテキストです。")
    dummy_ids = list(dummy_raw)
    if probe["add_bos"]:
        dummy_ids = [sp.bos_id()] + dummy_ids
    if probe["add_eos"]:
        dummy_ids = dummy_ids + [sp.eos_id()]
    dummy_inputs = {
        "input_ids": torch.tensor([dummy_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(dummy_ids)), dtype=torch.long),
    }
    with torch.no_grad():
        embedding_dim = int(wrapper(dummy_inputs["input_ids"], dummy_inputs["attention_mask"]).shape[-1])
    print(f"  埋め込み次元数: {embedding_dim}")

    fp32_onnx_path = os.path.join(out_dir, "model.onnx")
    cleanup_external_data(fp32_onnx_path)

    torch.onnx.export(
        wrapper,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
        external_data=False,
    )

    # PyTorch 2.14 Split 属性サニタイズ (単一 ONNX ファイルとして保存し外部データ参照を禁止)
    fp32_model = onnx.load(fp32_onnx_path)
    for node in fp32_model.graph.node:
        if node.op_type == "Split":
            attrs_to_remove = [a for a in node.attribute if a.name == "num_outputs"]
            for a in attrs_to_remove:
                node.attribute.remove(a)
    onnx.save(fp32_model, fp32_onnx_path, save_as_external_data=False)
    verify_inline_onnx(fp32_onnx_path)
    print(f"FP32 ONNX ready: {fp32_onnx_path}")

    # 4. FP16 変換
    print(f"[{model_key}] 4/5: FP16 ONNX 変換")
    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,
        disable_shape_infer=False
    )
    fixed_casts = sanitize_fp16_cast_nodes(fp16_model)
    if fixed_casts:
        print(f"  FP16 Cast ノード属性を {fixed_casts} 件補正しました")
    fp16_onnx_path = os.path.join(out_dir, "model_fp16.onnx")
    cleanup_external_data(fp16_onnx_path)
    onnx.save(fp16_model, fp16_onnx_path, save_as_external_data=False)
    verify_inline_onnx(fp16_onnx_path)
    print(f"FP16 ONNX ready: {fp16_onnx_path} ({os.path.getsize(fp16_onnx_path):,} bytes)")

    # 5. lite_config.json・ラッパー・README の配置
    print(f"[{model_key}] 5/5: lite_config.json・ラッパーおよび README の配置")
    if permuted:
        lite_bos_id, lite_eos_id, lite_unk_id = sp.bos_id(), sp.eos_id(), sp.unk_id()
        lite_pad_id = new_pad_id
    else:
        lite_bos_id = tokenizer.bos_token_id
        lite_eos_id = tokenizer.eos_token_id
        lite_unk_id = tokenizer.unk_token_id
        lite_pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else sp.pad_id()

    lite_config = {
        "model_key": model_key,
        "base_model": repo_id,
        "family": cfg["family"],
        "spm_fb": cfg["fb_filename"],
        "onnx": "model.onnx",
        "onnx_fp16": "model_fp16.onnx",
        "bos_id": lite_bos_id,
        "eos_id": lite_eos_id,
        "pad_id": lite_pad_id,
        "unk_id": lite_unk_id,
        "add_bos": probe["add_bos"],
        "add_eos": probe["add_eos"],
        "query_prefix": cfg["query_prefix"],
        "passage_prefix": cfg["passage_prefix"],
        "pooling": cfg["pooling"],
        "embedding_dim": embedding_dim,
        "license": cfg["license"],
        "token_id_space": "sentencepiece",
    }
    max_seq_length = _fetch_max_seq_length(repo_id)
    if max_seq_length:
        lite_config["max_seq_length"] = max_seq_length
    if permuted:
        lite_config["id_space_note"] = (
            "HF 側 ID 空間と異なるため、埋め込み行列の行を SentencePiece "
            "ネイティブ ID 空間に置換し、末尾に pad 専用行を追加済み。position_ids は新しい pad id で明示計算するよう調整済み"
        )
    with open(os.path.join(out_dir, "lite_config.json"), "w", encoding="utf-8") as f:
        json.dump(lite_config, f, ensure_ascii=False, indent=2)
    print(f"  lite_config.json: bos={lite_bos_id} eos={lite_eos_id} pad={lite_pad_id} unk={lite_unk_id}")

    loader_src = os.path.join(BASE_DIST_DIR, "ruri_v3_lite.py")
    if cfg["family"] == "ruri-v3":
        shutil.copyfile(loader_src, os.path.join(out_dir, "ruri_v3_lite.py"))
        readme_content = build_ruri_readme(model_key, cfg, wheel_path)
        # LICENSE ファイルの配置
        shutil.copyfile("LICENSE", os.path.join(out_dir, "LICENSE"))
    else:
        shutil.copyfile(loader_src, os.path.join(out_dir, "lite_embed.py"))
        readme_content = build_generic_readme(model_key, cfg, wheel_path, lite_config, permuted)
        write_license_file(cfg, out_dir)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✨ {repo_id} ({model_key}) の資産生成が完了しました: {out_dir}")


def build_ruri_readme(model_key: str, cfg: dict, wheel_path: str) -> str:
    repo_id = cfg["repo_id"]
    return f"""---
license: apache-2.0
language:
- ja
base_model: {repo_id}
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- embeddings
- sentencepiece_lite
- onnx
- zero-torch
---

# ruri-v3-{model_key}-lite: Zero-Torch & Ultra-Fast Japanese Embeddings

[`{repo_id}`](https://huggingface.co/{repo_id}) を、**PyTorch / Transformers 一切不要（Zero-Torch）** かつ **C++ コンパイル不要（事前ビルドWheel配布）** で実行できるように最適化した軽量日本語埋め込み環境です。

## インストール手順

```bash
pip install https://huggingface.co/Chottokun/ruri-v3-{model_key}-lite/resolve/main/wheels/{os.path.basename(wheel_path)} onnxruntime numpy huggingface_hub
```

## クイックスタート

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-{model_key}-lite")

queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

q_emb = model.encode(queries)
d_emb = model.encode(documents)

similarities = np.dot(q_emb, d_emb.T)[0]
print(f"クエリ vs 東京: {{similarities[0]:.4f}}")
print(f"クエリ vs 大阪: {{similarities[1]:.4f}}")
```

---

## ⚖️ License & Redistribution Terms (Apache License, Version 2.0)

This model repository, converted binaries, and wrapper scripts are distributed under the **Apache License, Version 2.0** (the "License"). You may not use these files except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Upstream Components & Notices of Modification (Section 4)

1. **Base Embedding Model**:
   - **Model**: [{repo_id}](https://huggingface.co/{repo_id})
   - **Copyright**: Copyright 2024 Hayato Tsukagoshi and Ryohei Sasano (Nagoya University NLP Laboratory)
   - **License**: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
   - **Prominent Notice of Modification (Section 4b)**:
     - Converted original PyTorch Safetensors weights into ONNX (FP32) and FP16 formats.
     - Converted official SentencePiece model into FlatBuffers serialization format (`.spm.fb`).
     - Added independent inference implementation (`ruri_v3_lite.py`) decoupling from PyTorch and Transformers.

2. **Tokenizer Engine**:
   - **Library**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
   - **Repository**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
   - **Copyright**: Copyright 2018 Google LLC
   - **License**: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
   - **Prominent Notice of Modification (Section 4b)**:
     - Built and packaged as a standalone C++20 Python wheel with Safe Boundary Pre-tokenization (SBP) support.

### 📚 Citations

#### Ruri (Embedding Model)
```bibtex
@misc{{Ruri,
  title={{Ruri: Japanese General Text Embeddings}}, 
  author={{Hayato Tsukagoshi and Ryohei Sasano}},
  year={{2024}},
  eprint={{2409.07737}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}},
  url={{https://arxiv.org/abs/2409.07737}}, 
}}
```

#### SentencePiece (Tokenizer)
```bibtex
@inproceedings{{kudo-richardson-2018-sentencepiece,
  title = "{{S}}entence{{P}}iece: A simple and language independent subword tokenizer and detokenizer for {{N}}eural {{T}}ext {{P}}rocessing",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  month = nov,
  year = "2018",
  address = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D18-2012",
  doi = "10.18653/v1/D18-2012",
  pages = "66--71",
}}
```

---

## 📄 License File
A full copy of the license is included in [LICENSE](LICENSE).
"""


# モデルごとの上流クレジット情報 (汎用 README / LICENSE 生成用)
UPSTREAM_CREDITS = {
    "multilingual-e5": {
        "copyright": "Copyright (c) Microsoft Corporation",
        "license_url": "https://opensource.org/licenses/MIT",
        "citation": """```bibtex
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
```""",
    },
    "embeddinggemma": {
        "copyright": "Copyright 2025 Google LLC",
        "license_url": "http://www.apache.org/licenses/LICENSE-2.0",
        "citation": "",
    },
}


def write_license_file(cfg: dict, out_dir: str) -> None:
    """上流モデルのライセンス本文を LICENSE として配置する"""
    credits = UPSTREAM_CREDITS.get(cfg["family"], {})
    if cfg["license"] == "mit":
        text = f"""MIT License

{credits.get('copyright', '')}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        with open(os.path.join(out_dir, "LICENSE"), "w", encoding="utf-8") as f:
            f.write(text)
    else:
        # Apache-2.0: リポジトリ同梱のライセンス本文をそのまま使う
        shutil.copyfile("LICENSE", os.path.join(out_dir, "LICENSE"))


def build_generic_readme(model_key: str, cfg: dict, wheel_path: str, lite_config: dict, permuted: bool) -> str:
    repo_id = cfg["repo_id"]
    display = cfg["display_name"]
    credits = UPSTREAM_CREDITS.get(cfg["family"], {})
    license_name = "MIT License" if cfg["license"] == "mit" else "Apache License, Version 2.0"

    modifications = [
        "Converted original PyTorch Safetensors weights into ONNX (FP32) and FP16 formats (single-file, no external weight data).",
        "Converted official SentencePiece model into FlatBuffers serialization format (`.spm.fb`).",
        "Added independent inference implementation (`lite_embed.py`) decoupling from PyTorch and Transformers.",
        "Special token IDs and BOS/EOS behavior were measured from the upstream tokenizer at build time and recorded in `lite_config.json` (no hardcoding).",
    ]
    if cfg["export_target"] == "language_model":
        modifications.append(
            "Exported the text tower (`language_model`) only; its outputs on text inputs are identical to the full multimodal model."
        )
    if permuted:
        modifications.append(
            "Permuted the token embedding matrix rows into the SentencePiece-native ID space "
            "(upstream HF token IDs differ from SentencePiece IDs for this model) and appended a dedicated padding row. Position IDs are computed explicitly with the new padding ID so position semantics match the upstream model."
        )
    mod_lines = "\n".join(f"     - {m}" for m in modifications)
    citation = credits.get("citation", "")
    citation_block = f"\n### 📚 Citation\n\n{citation}\n" if citation else ""

    return f"""---
license: {cfg['license']}
base_model: {repo_id}
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- embeddings
- sentencepiece_lite
- onnx
- zero-torch
---

# {display}-lite: Zero-Torch Embeddings with sentencepiece_lite

[`{repo_id}`](https://huggingface.co/{repo_id}) を、**PyTorch / Transformers 一切不要（Zero-Torch）** かつ **C++ コンパイル不要（事前ビルドWheel配布）** で実行できる軽量埋め込み環境です。トークナイズは SentencePiece の FlatBuffers 形式 (`.spm.fb`) を `sentencepiece_lite` で処理し、推論は ONNX Runtime で行います。

## インストール手順

```bash
pip install wheels/{os.path.basename(wheel_path)} onnxruntime numpy huggingface_hub
```

## クイックスタート

```python
from lite_embed import LiteEmbedder
import numpy as np

model = LiteEmbedder(model_dir=".")  # このディレクトリを指定

q_emb = model.encode_query(["日本の首都はどこですか？"])
d_emb = model.encode_passages([
    "日本の首都は東京都です。",
    "大阪は関西地方の主要都市です。",
])

similarities = np.dot(q_emb, d_emb.T)[0]
print(f"クエリ vs 東京: {{similarities[0]:.4f}}")
print(f"クエリ vs 大阪: {{similarities[1]:.4f}}")
```

- クエリ接頭辞: `{cfg['query_prefix']}`
- 文書接頭辞: `{cfg['passage_prefix']}`
- 特殊トークン (BOS={lite_config['bos_id']}, EOS={lite_config['eos_id']}, PAD={lite_config['pad_id']}) と付与条件は `lite_config.json` に記録されています。

---

## ⚖️ License

- **Base Embedding Model**: [{repo_id}](https://huggingface.co/{repo_id})
  - {credits.get('copyright', '')}
  - License: [{license_name}]({credits.get('license_url', '')})
- **Tokenizer Engine**: [Google SentencePiece Lite](https://github.com/google/sentencepiece) — Copyright 2018 Google LLC, Apache License, Version 2.0. Built and packaged as a standalone C++20 Python wheel with Safe Boundary Pre-tokenization (SBP) support.

### Notices of Modification

1. **Base Embedding Model**:
   - **Prominent Notice of Modification**:
{mod_lines}

A full copy of the base model's license is included in [LICENSE](LICENSE).
{citation_block}
#### SentencePiece (Tokenizer)
```bibtex
@inproceedings{{kudo-richardson-2018-sentencepiece,
  title = "{{S}}entence{{P}}iece: A simple and language independent subword tokenizer and detokenizer for {{N}}eural {{T}}ext {{P}}rocessing",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  month = nov,
  year = "2018",
  address = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D18-2012",
  doi = "10.18653/v1/D18-2012",
  pages = "66--71",
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="lite 資産生成スクリプト (ruri-v3 / multilingual-e5 / embeddinggemma)")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="70m", help="対象モデル")
    parser.add_argument("--all", action="store_true", help="ruri-v3 の 70m/130m/310m を一括生成 (従来互換)")
    parser.add_argument("--out-root", default=None, help="出力先ルート (既定: dist_assets)")
    args = parser.parse_args()

    spm_to_fb_bin, wheel_path = ensure_sentencepiece_tools_and_wheel()

    if args.all:
        for k in ["70m", "130m", "310m"]:
            export_model(k, spm_to_fb_bin, wheel_path, out_root=args.out_root)
    else:
        export_model(args.model, spm_to_fb_bin, wheel_path, out_root=args.out_root)

if __name__ == "__main__":
    main()
