"""Hugging Face デプロイスクリプト (マルチモデル対応版)
指定したモデル (ruri-v3: 30m/70m/130m/310m, e5-base, gemma2) または
ruri-v3 全モデルをデプロイする。

使用例:
  uv run --with "huggingface_hub" python scripts/deploy_to_hf.py --model 70m
  uv run --with "huggingface_hub" python scripts/deploy_to_hf.py --model e5-base
  uv run --with "huggingface_hub" python scripts/deploy_to_hf.py --model gemma2 --out-root ~/lite_factory_outputs
"""
import os
import sys
import argparse
from huggingface_hub import HfApi, get_token

BASE_DIST_DIR = os.path.abspath("./dist_assets")

# build_and_export.py への依存 (torch/transformers) を避けるため、デプロイ用設定は独立して持つ
DEPLOY_CONFIGS = {
    "30m": {"repo_suffix": "ruri-v3-30m-lite", "dirname": None},
    "70m": {"repo_suffix": "ruri-v3-70m-lite", "dirname": "ruri_v3_70m"},
    "130m": {"repo_suffix": "ruri-v3-130m-lite", "dirname": "ruri_v3_130m"},
    "310m": {"repo_suffix": "ruri-v3-310m-lite", "dirname": "ruri_v3_310m"},
    "e5-base": {"repo_suffix": "multilingual-e5-base-lite", "dirname": "multilingual_e5_base"},
    "gemma2": {"repo_suffix": "embeddinggemma-2-lite", "dirname": "embeddinggemma_2"},
}

OTHER_MODEL_DIRS = [c["dirname"] for c in DEPLOY_CONFIGS.values() if c["dirname"]]


def deploy_model(model_key: str, api: HfApi, username: str, out_root: str = None):
    cfg = DEPLOY_CONFIGS[model_key]
    repo_id = f"{username}/{cfg['repo_suffix']}"

    if out_root:
        base_dir = os.path.abspath(os.path.expanduser(out_root))
        target_dir = os.path.join(base_dir, cfg["dirname"] or "ruri_v3_30m")
    else:
        target_dir = BASE_DIST_DIR if cfg["dirname"] is None else os.path.join(BASE_DIST_DIR, cfg["dirname"])

    if not os.path.exists(target_dir):
        print(f"エラー: デプロイ対象ディレクトリが存在しません: {target_dir}")
        out_opt = f" --out-root {out_root}" if out_root else ""
        print(f"先に `uv run python scripts/build_and_export.py --model {model_key}{out_opt}` を実行してください。")
        return False

    print(f"\n==================================================")
    print(f"  デプロイ開始: {repo_id}")
    print(f"  送信元ディレクトリ: {target_dir}")
    print(f"==================================================")

    print(f"=== [1/3] リポジトリ存在確認・作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"リポジトリ準備完了: https://huggingface.co/{repo_id}")

    print(f"\n=== [2/3] アップロード対象ファイルの確認 ===")
    files_to_upload = []
    for root, dirs, files in os.walk(target_dir):
        # サブディレクトリとしての他モデルディレクトリやキャッシュは除外
        rel_root = os.path.relpath(root, target_dir)
        top_dir = rel_root.split(os.sep, 1)[0]
        if (
            rel_root.startswith("ruri_v3_")
            or top_dir in OTHER_MODEL_DIRS
            or "__pycache__" in rel_root
        ):
            continue
        for f in files:
            if (
                f.endswith(".pyc")
                or f.endswith(".onnx.data")
                or f.endswith(".onnx_data")
                or f.startswith("sentencepiece_lite-0.1.0-")
                or f == "model_fixed.onnx"
                or f == "ruri_v3_reranker_lite.py"
            ):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, target_dir)
            files_to_upload.append(rel_path)
            print(f"  - {rel_path} ({os.path.getsize(full_path):,} bytes)")

    print(f"\n=== [3/3] 成果物の一括アップロード開始 ===")
    ignore_patterns = [
        "ruri_v3_*/**",
        *[f"{d}/**" for d in OTHER_MODEL_DIRS if not d.startswith("ruri_v3_")],
        "__pycache__/**",
        "**/__pycache__/**",
        "*.pyc",
        "model_fixed.onnx",
        "ruri_v3_reranker_lite.py",
        "*.onnx.data",
        "*.onnx_data",
        "wheels/sentencepiece_lite-0.1.0-*.whl",
    ]
    api.upload_folder(
        folder_path=target_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: deploy {cfg['repo_suffix']} assets (sentencepiece_lite cp311-abi3)",
        ignore_patterns=ignore_patterns,
        delete_patterns=[
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
            "*.onnx.data",
            "*.onnx_data",
            "model_fixed.onnx",
            "wheels/sentencepiece_lite-0.1.0-*.whl",
        ],
    )
    print(f"\n【デプロイ完了】{repo_id} のアップロードが完了しました！ 🎉")
    print(f"URL: https://huggingface.co/{repo_id}")
    return True

def load_env_file():
    """ローカルの .env ファイルが存在する場合に環境変数を読み込む (サードパーティ非依存)"""
    env_path = os.path.abspath("./.env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and v and k not in os.environ:
                        os.environ[k] = v

def main():
    load_env_file()

    parser = argparse.ArgumentParser(description="Hugging Face デプロイスクリプト")
    parser.add_argument("--model", choices=list(DEPLOY_CONFIGS.keys()), default="70m", help="デプロイ対象モデル")
    parser.add_argument("--all", action="store_true", help="ruri-v3 の 30m/70m/130m/310m を一括デプロイ (従来互換)")
    parser.add_argument("--out-root", default=None, help="ビルド成果物のルートディレクトリ (既定: dist_assets)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        print("エラー: 有効な Hugging Face トークンが見つかりません。")
        print("設定方法:")
        print("  1. .env ファイルに `HF_TOKEN=hf_xxx` を記述")
        print("  2. または環境変数 `export HF_TOKEN=hf_xxx` を設定")
        sys.exit(1)

    api = HfApi(token=hf_token)
    
    # ユーザー名: HF_USERNAME 環境変数があれば最優先、なければトークンから自動取得
    username = os.environ.get("HF_USERNAME")
    if not username:
        user_info = api.whoami()
        username = user_info["name"]

    if args.all:
        for m in ["30m", "70m", "130m", "310m"]:
            deploy_model(m, api, username, out_root=args.out_root)
    else:
        deploy_model(args.model, api, username, out_root=args.out_root)

if __name__ == "__main__":
    main()
