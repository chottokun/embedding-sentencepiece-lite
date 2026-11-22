"""ruri-v3-reranker-310m-lite の Hugging Face Hub アップロードスクリプト
dist_assets/ruri_v3_reranker_310m 内のすべての資材を Chottokun/ruri-v3-reranker-310m-lite にアップロードします。
"""
import os
import sys
from huggingface_hub import HfApi, create_repo

SRC_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

def load_env():
    env_file = os.path.abspath(".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def main():
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("エラー: HF_TOKEN が設定されていません。.env を確認してください。")
        sys.exit(1)

    username = os.environ.get("HF_USERNAME", "Chottokun")
    repo_id = f"{username}/ruri-v3-reranker-310m-lite"

    print("=" * 70)
    print(f"  Hugging Face Hub アップロード開始: {repo_id}")
    print(f"  ソースディレクトリ: {SRC_DIR}")
    print("=" * 70)

    api = HfApi(token=token)

    # リポジトリ作成 (存在しない場合)
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    print(f"リポジトリ確認完了: https://huggingface.co/{repo_id}")

    # フォルダ全体のアップロード (不要な .onnx.data / 中間ファイルを除外し、リモートの残存 .onnx.data も削除)
    print("資材をアップロード中 (ONNX, FlatBuffers, Wheels, ラッパー, ドキュメント)...")
    api.upload_folder(
        folder_path=SRC_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="chore: remove orphaned .onnx.data and update ruri-v3-reranker-310m assets",
        ignore_patterns=[
            "__pycache__/**",
            "*.pyc",
            "*.onnx.data",
            "*.onnx_data",
            "model_clean.onnx",
            "model_pruned_16l.onnx",
            "model_pruned_16l-inferred.onnx",
        ],
        delete_patterns=["*.onnx.data", "*.onnx_data"],
    )

    print("\n" + "=" * 70)
    print(f"🎉 アップロード完了！")
    print(f"URL: https://huggingface.co/{repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
