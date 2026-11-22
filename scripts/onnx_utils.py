"""ONNX エクスポート時の外部データ残存防止・検証ユーティリティ"""
import os
import onnx


def cleanup_external_data(onnx_path: str) -> None:
    """不要な外部重みファイル (*.onnx.data / *.onnx_data) が残存していれば削除する"""
    for suffix in (".data", "_data"):
        data_path = onnx_path + suffix
        if os.path.exists(data_path):
            os.remove(data_path)
            print(f"不要な外部データファイルを削除しました: {data_path}")


def verify_inline_onnx(onnx_path: str) -> None:
    """ONNX ファイルが外部データ参照を持たない単一ファイルであることを検証する"""
    model_proto = onnx.load(onnx_path, load_external_data=False)
    external_tensors = [
        t.name for t in model_proto.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external_tensors:
        raise RuntimeError(
            f"{onnx_path} に外部データ参照が {len(external_tensors)} 件残っています: {external_tensors[:5]}"
        )
    cleanup_external_data(onnx_path)
