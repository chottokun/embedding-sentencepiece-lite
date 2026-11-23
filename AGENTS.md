# AGENTS.md

## Strict Constraints
- **Package Manager**: All executions and installations must use `uv`. Direct usage of `pip` or `conda` is forbidden.
- **Runtime Isolation**:
  - `dist_assets/ruri_v3_lite.py` must NEVER import `torch` or `transformers`.
  - Allowed runtime dependencies: `sentencepiece_lite`, `onnxruntime` / `onnxruntime-gpu`, `numpy`, `huggingface_hub`.
- **Inference Separation**:
  - CPU: Tokenization via `sentencepiece_lite` (BOS/EOS injection, padding, truncation).
  - GPU/CPU: Matrix multiplication via `onnxruntime`.
- **Model Switching & Precision Policy**:
  - Check CUDA Compute Capability via `ctypes` (`nvcuda.dll` / `libcuda.so`). Do NOT import PyTorch for this check.
  - Capability < 7.0 (e.g. GTX 1080) or CPU: Load `model.onnx` (FP32).
  - Capability >= 7.0 (e.g. Tesla T4, RTX 3060): Load `model_fp16.onnx` (FP16) by default when `allow_fp16=true` (`gpu_precision="fp16"`).
  - Model-specific override: When `lite_config.json` sets `"gpu_precision": "fp32"` / `"allow_fp16": false` (e.g. `embeddinggemma-2`, whose activation dynamic range degrades in IEEE FP16 on GPU), load `model.onnx` (FP32) even on Capability >= 7.0 GPUs.
- **Security & Secret Hygiene (GitHub Public Readiness)**:
  - **No Hardcoded Tokens/Secrets**: NEVER commit API tokens, Hugging Face write tokens (`HF_TOKEN`), credentials, or private file paths into any file.
  - Token input must exclusively come from environment variables (e.g. `os.environ.get("HF_TOKEN")`) or user prompt.
  - Large binaries (`*.onnx`, `*.spm.fb`, `*.safetensors`, `.whl`) and cache directories (`.venv/`, `.cache/`, `hf_cache/`) must NOT be committed to git (ensure `.gitignore` exclusion).
  - Clean local logs and scratchpads before committing. Verify `git status` / `git diff` before every commit.

---

## File Structure & Roles
- `packages/sentencepiece_lite/`: Source code for the lightweight sentencepiece_lite C++/Python extension with FastSBPTokenizer (CPython Stable ABI `cp311-abi3`).
- `.github/workflows/ci.yml`: Automated CI pipeline (cp311-abi3 wheel build, `abi3audit`, Python 3.11–3.13 matrix tests, `pip-audit`/`bandit`/`cppcheck`/`gitleaks`).
- `.github/workflows/release.yml`: Automated GitHub Releases workflow triggered by `v*` tags (`cp311-abi3` wheel + `SHA256SUMS.txt`).
- `scripts/verify_tokenizer.py`: Phase 0 verification to confirm BOS/EOS token IDs, special tokens, and pooling behavior.
- `scripts/export_tokenizer_json.py`: Export HF Fast Tokenizer format (tokenizer.json, tokenizer_config.json) with verified special tokens.
- `scripts/build_and_export.py`: Build `.whl` from source, export FP32 ONNX, and convert to FP16. (PyTorch/Transformers allowed here).
- `scripts/verify_lite_set.py`: Verify generated lite set (tokenizer IDs, FP32 cosine similarity, FP16 route & GPU precision policy gate, and `HF_HUB_OFFLINE=1` offline smoke test).
- `scripts/deploy_to_hf.py`: Upload `dist_assets/` to target Hugging Face model repo.
- `scripts/test_inference.py`: Verification script executing `dist_assets/ruri_v3_lite.py` under various text inputs.
- `dist_assets/ruri_v3_lite.py`: Self-contained wrapper class for end-users.
- `dist_assets/wheels/`: Built `sentencepiece_lite` wheel files.

---

## Command Reference (uv)

### 1. Environment Setup
```bash
uv init --app
uv venv --python 3.11
```

### 2. Phase 0 Verification
```bash
uv run --with "torch" --with "transformers" --with "sentencepiece" --with "huggingface_hub" python scripts/verify_tokenizer.py
```

### 2.5 Fast Tokenizer Export (tokenizer.json)
```bash
uv run --with "transformers" --with "sentencepiece" python scripts/export_tokenizer_json.py
```

### 3. Asset Build & Export
```bash
# PyTorch 2.14+ ONNX exporter requires onnxscript and sentencepiece
uv run --with "torch" --with "transformers" --with "onnx" --with "onnxscript" --with "onnxconverter-common" --with "sentencepiece" --with "build" --with "huggingface_hub" python scripts/build_and_export.py --model 70m

# For embeddinggemma-2 (requires latest transformers with embedding_gemma2 support):
uv run --with "git+https://github.com/huggingface/transformers.git" --with "torch" --with "onnx" --with "onnxscript" --with "onnxconverter-common" --with "sentencepiece" --with "build" --with "huggingface_hub" python scripts/build_and_export.py --model gemma2
```

### 3.5 Lite Set Verification (Tokenizer + FP32/FP16 Gate + Offline Smoke)
```bash
uv run --with "torch" --with "transformers" --with "sentence-transformers" --with "sentencepiece" python scripts/verify_lite_set.py --model-dir ./dist_assets/multilingual_e5_base --base-model intfloat/multilingual-e5-base
```

### 4. Deploy to Hugging Face
```bash
export HF_TOKEN="hf_xxx" # Pass via environment variable, never write in script
uv run --with "huggingface_hub" python scripts/deploy_to_hf.py
```

### 5. Lightweight Verification (No PyTorch)
```bash
# CPU test
uv run --with "onnxruntime" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py

# GPU test
uv run --with "onnxruntime-gpu" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py
```