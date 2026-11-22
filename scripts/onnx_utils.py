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


def sanitize_fp16_cast_nodes(fp16_model: onnx.ModelProto) -> int:
    """FP16 変換後に value_info が FLOAT16 なのに to 属性が FLOAT のまま残った Cast ノードを補正する"""
    vi_types = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(fp16_model.graph.value_info)
        + list(fp16_model.graph.output)
        + list(fp16_model.graph.input)
    }
    fixed_count = 0
    for node in fp16_model.graph.node:
        if node.op_type == "Cast" and node.output:
            if vi_types.get(node.output[0]) == onnx.TensorProto.FLOAT16:
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                        attr.i = onnx.TensorProto.FLOAT16
                        fixed_count += 1
    return fixed_count


def verify_inline_onnx(onnx_path: str) -> None:
    """ONNX ファイルが外部データ参照を持たず、型・形状チェックにも合格することを検証する"""
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
    onnx.checker.check_model(onnx_path, full_check=True)
