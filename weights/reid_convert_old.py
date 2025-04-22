import onnx
from coram.tensorrt import onnx2trt, save_trt_engine

save_trt_path = "1635581407_43eph.engine"

onnx_model_path = "1635581407_43eph.onnx"
onnx_model = onnx.load(onnx_model_path)


opt_shape_dict = {
    "input": [
        [1, 3, 448, 224],
        [8, 3, 448, 224],
        [16, 3, 448, 224],
    ],
}

max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    mode=1,
    max_workspace_size=max_workspace_size,
)

save_trt_engine(trt_engine, save_trt_path)