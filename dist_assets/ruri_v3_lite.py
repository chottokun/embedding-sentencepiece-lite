"""lite 埋め込み推論ラッパーモジュール (ruri-v3 初出・モデル汎用版)
PyTorch / Transformers は完全不要。
依存パッケージ: sentencepiece_lite, onnxruntime / onnxruntime-gpu, numpy, huggingface_hub

モデル固有の情報 (特殊トークン ID・BOS/EOS 付与の有無・接頭辞・プーリング方式・
.spm.fb ファイル名) は、モデルディレクトリ内の lite_config.json から読み込む。
lite_config.json が存在しない場合は ruri-v3 の従来動作
(BOS=1 / EOS=2 / PAD=3、BOS+EOS 付与、Mean Pooling) にフォールバックするため、
既存の ruri-v3-*-lite リポジトリでもそのまま動作する。

ファイル解決はローカル優先: 明示指定の model_dir があればそのディレクトリのみ、
未指定ならローダー自身のディレクトリを探し、ローカルにあれば Hugging Face Hub の
API は一切呼ばない (HF_HUB_OFFLINE=1 でもそのまま動作する)。
ローカルに無いファイルだけを Hub から取得する。
"""
import ctypes
import json
import os
from typing import List, Union
import numpy as np
import onnxruntime as ort
import sentencepiece_lite as spl
from huggingface_hub import hf_hub_download

# lite_config.json が無い場合のフォールバック (ruri-v3 の確定済み特殊トークン ID)
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>
PAD_ID = 3   # <pad>

DEFAULT_CONFIG = {
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
    "pad_id": PAD_ID,
    "add_bos": True,
    "add_eos": True,
    "query_prefix": "検索クエリ: ",
    "passage_prefix": "文章: ",
    "pooling": "mean",
    "spm_fb": None,        # None の場合は従来の ruri_v3_<size>.spm.fb 命名規則で解決
    "embedding_dim": None,
    "gpu_precision": "fp16",   # "fp16" (CC>=7.0でFP16選択) | "fp32" (GPUでもFP32固定)
    "allow_fp16": True,
    "fp16_cos_threshold": 0.995,
}

# モデルサイズ別の埋め込み次元数フォールバックマップ (ruri-v3 用)
MODEL_DIMS = {"30m": 256, "70m": 384, "130m": 512, "310m": 768}

def get_cuda_compute_capability() -> float:
    """CUDA ドライバ経由で GPU の Compute Capability を取得 (PyTorch 完全不要)"""
    try:
        cuda = ctypes.CDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so")
        if cuda.cuInit(0) != 0:
            return 0.0
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return 0.0

        major = ctypes.c_int()
        minor = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
        return major.value + (minor.value / 10.0)
    except Exception:
        return 0.0

def _local_search_dirs(model_dir: str = None) -> List[str]:
    """モデルファイルのローカル探索先: 明示指定があればそのディレクトリのみ、未指定ならローダー自身のディレクトリ"""
    if model_dir:
        return [model_dir]
    return [os.path.dirname(os.path.abspath(__file__))]


class RuriV3Lite:
    """lite 埋め込み軽量推論クラス (lite_config.json 駆動・モデル汎用)"""

    @staticmethod
    def _find_local(model_dir: str, filename: str) -> Union[str, None]:
        """ローカル探索先からファイルを探し、存在すればそのパスを返す (Hub は呼ばない)"""
        for d in _local_search_dirs(model_dir):
            candidate = os.path.join(d, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _resolve_model_path(repo_id: str, model_dir: str, filename: str) -> str:
        """モデルファイルパスを解決する (ローカル優先・ローカルに無ければ Hub から取得)"""
        local = RuriV3Lite._find_local(model_dir, filename)
        if local:
            return local
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _load_config(self, repo_id: str, model_dir: str) -> dict:
        """lite_config.json を読み込み、DEFAULT_CONFIG とマージして返す"""
        config = dict(DEFAULT_CONFIG)
        cfg_path = self._find_local(model_dir, "lite_config.json")
        if cfg_path is None and not (model_dir and os.path.exists(model_dir)):
            # 明示ディレクトリが指定されていない場合のみ Hub へ問い合わせる
            # (ruri 従来動作の維持)。失敗すれば既定値フォールバック。
            try:
                cfg_path = hf_hub_download(repo_id=repo_id, filename="lite_config.json")
            except Exception:
                cfg_path = None
        if cfg_path:
            with open(cfg_path, encoding="utf-8") as f:
                loaded = json.load(f)
            config.update({k: v for k, v in loaded.items() if v is not None})
            print(f"[RuriV3Lite] lite_config.json を読み込みました: {cfg_path}")
        return config

    def __init__(
        self,
        repo_id: str = "Chottokun/ruri-v3-30m-lite",
        model_dir: str = None,
        force_fp32: bool = False,
        force_fp16: bool = False,
        device: str = "auto"  # "auto", "cuda", "cpu"
    ):
        """
        初期化:
        model_dir が指定されている場合はローカルパスから読み込み、
        未指定の場合はローダー自身のディレクトリ、次に Hugging Face Hub (repo_id) から取得します。
        """
        # 0. モデル固有設定 (特殊トークン・接頭辞・プーリング・精度ポリシー等) の読み込み
        self.config = self._load_config(repo_id, model_dir)
        self.bos_id = int(self.config["bos_id"])
        self.eos_id = int(self.config["eos_id"])
        self.pad_id = int(self.config["pad_id"])
        self.add_bos = bool(self.config["add_bos"])
        self.add_eos = bool(self.config["add_eos"])
        self.allow_fp16 = bool(self.config.get("allow_fp16", True)) and (
            self.config.get("gpu_precision", "fp16") == "fp16"
        )

        available_providers = ort.get_available_providers()
        want_cuda = (device == "cuda") or (device == "auto" and "CUDAExecutionProvider" in available_providers)

        onnx_fp32_name = self.config.get("onnx") or "model.onnx"
        onnx_fp16_name = self.config.get("onnx_fp16") or "model_fp16.onnx"

        # 1. ハードウェア世代およびモデル別精度ポリシーに応じたモデル判定
        if force_fp16 and not force_fp32:
            model_filename = onnx_fp16_name
            print("[RuriV3Lite] force_fp16=True -> FP16 モデルをロード")
        elif want_cuda and not force_fp32 and self.allow_fp16:
            capability = get_cuda_compute_capability()
            # Turing (7.5) / Ampere (8.6: RTX 3060) 以降は FP16
            # Pascal (6.1: GTX 1080) 等は 7.0 未満のため FP32
            if capability >= 7.0:
                model_filename = onnx_fp16_name
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} >= 7.0 検知 -> FP16 モデルをロード")
            else:
                model_filename = onnx_fp32_name
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} < 7.0 検知 (GTX 1080等) -> FP32 モデルをロード")
        elif want_cuda and not force_fp32 and not self.allow_fp16:
            model_filename = onnx_fp32_name
            print("[RuriV3Lite] モデル精度ポリシー (gpu_precision=fp32) に従い FP32 モデルをロード")
        else:
            model_filename = onnx_fp32_name
            print("[RuriV3Lite] CPU 実行または force_fp32=True -> FP32 モデルをロード")

        self.model_filename = model_filename

        # 2. モデル & トークナイザーの取得
        model_size = "30m"
        for s in ["310m", "130m", "70m", "30m"]:
            if s in repo_id.lower():
                model_size = s
                break

        fb_filename = self.config.get("spm_fb") or f"ruri_v3_{model_size}.spm.fb"

        model_path = self._resolve_model_path(repo_id, model_dir, model_filename)
        fb_path = self._find_local(model_dir, fb_filename)
        if fb_path is None:
            for d in _local_search_dirs(model_dir):
                if os.path.isdir(d):
                    fbs = sorted(f for f in os.listdir(d) if f.endswith(".spm.fb"))
                    if fbs:
                        fb_path = os.path.join(d, fbs[0])
                        break
        if fb_path is None:
            try:
                fb_path = hf_hub_download(repo_id=repo_id, filename=fb_filename)
            except Exception:
                fb_path = hf_hub_download(repo_id=repo_id, filename="ruri_v3_30m.spm.fb")

        # 3. SentencePiece Lite トークナイザーの初期化 (FlatBuffers, SBP並列対応)
        self.tokenizer = spl.FastSBPTokenizer(fb_path)

        # 4. ONNX Runtime セッション初期化 (CUDA 失敗時は FP32 に切り替えて CPU へフォールバック)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if want_cuda:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
            try:
                self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
                # 実際に CUDA が選ばれたか確認
                active_provider = self.session.get_providers()[0]
                print(f"[RuriV3Lite] InferenceSession 初期化完了 (プロバイダ: {active_provider})")
            except Exception as e:
                print(f"[RuriV3Lite] CUDA 初期化失敗 ({e}) -> CPUExecutionProvider にフォールバックします")
                if not force_fp16 and model_filename != onnx_fp32_name:
                    model_filename = onnx_fp32_name
                    self.model_filename = model_filename
                    model_path = self._resolve_model_path(repo_id, model_dir, model_filename)
                self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
            print("[RuriV3Lite] InferenceSession 初期化完了 (プロバイダ: CPUExecutionProvider)")

        # 出力テンソル形状から埋め込み次元数を取得 (動的軸等で非 int の場合は設定/マップからフォールバック)
        out_shape = self.session.get_outputs()[0].shape
        last_dim = out_shape[-1] if out_shape else None
        if isinstance(last_dim, int):
            self.embedding_dim = last_dim
        elif self.config.get("embedding_dim"):
            self.embedding_dim = int(self.config["embedding_dim"])
        else:
            self.embedding_dim = MODEL_DIMS.get(model_size, 256)

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        テキストのリストを高次元埋め込みベクトルに変換します。

        Args:
            texts: 単一文字列または文字列のリスト
            max_length: 最大トークン長 (BOS/EOS含む)
            batch_size: バッチサイズ
            normalize_embeddings: True の場合 L2 正規化を実行 (コサイン類似度用)

        Returns:
            np.ndarray: shape (len(texts), self.embedding_dim) の埋め込みベクトル
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        pooling = self.config.get("pooling", "mean")
        if pooling != "mean":
            raise ValueError(f"未対応のプーリング方式です: {pooling}")

        reserve = (1 if self.add_bos else 0) + (1 if self.add_eos else 0)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # CPU 上で sentencepiece_lite による高速トークナイズ
            batch_ids = []
            for text in batch_texts:
                tokens = self.tokenizer.encode(text)
                # BOS/EOS の付与は lite_config.json の設定に従う (モデル側定義が正)
                clipped_tokens = tokens[:max(0, max_length - reserve)]
                ids = ([self.bos_id] if self.add_bos else []) + clipped_tokens + ([self.eos_id] if self.add_eos else [])
                batch_ids.append(ids)

            seq_len = max(len(ids) for ids in batch_ids)
            # pad_id でパディング初期化
            input_ids = np.full((len(batch_texts), seq_len), fill_value=self.pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(batch_texts), seq_len), dtype=np.int64)

            for b_idx, ids in enumerate(batch_ids):
                input_ids[b_idx, :len(ids)] = ids
                attention_mask[b_idx, :len(ids)] = 1

            # ONNX Runtime 推論
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            # Mean Pooling (attention_mask を考慮した加重平均)
            hidden_states = outputs[0].astype(np.float32)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 正規化
            if normalize_embeddings:
                norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                mean_pooled = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)

            all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings)

    @staticmethod
    def _apply_prefix(texts: Union[str, List[str]], prefix: str) -> Union[str, List[str]]:
        if isinstance(texts, str):
            return prefix + texts
        return [prefix + t for t in texts]

    def encode_query(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """検索クエリ用: lite_config.json の query_prefix を付与して埋め込む"""
        return self.encode(self._apply_prefix(texts, self.config.get("query_prefix", "")), **kwargs)

    def encode_passages(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """検索対象文書用: lite_config.json の passage_prefix を付与して埋め込む"""
        return self.encode(self._apply_prefix(texts, self.config.get("passage_prefix", "")), **kwargs)

    # 文書側エイリアス
    encode_documents = encode_passages


# モデル汎用の別名 (ruri 以外の lite セットではこちらを使う)
LiteEmbedder = RuriV3Lite
