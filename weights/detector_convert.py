import onnx

from coram.tensorrt import onnx2trt, save_trt_engine

load_onnx = "/maydrive/detector_weights/yolox_x_pretrained.onnx"
onnx_model = onnx.load(load_onnx)

opt_shape_dict = {
    "input": [
        [1, 3, 448, 832],
        [8, 3, 448, 832],
        [16, 3, 448, 832],
    ],
}

max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    mode=1,
    max_workspace_size=max_workspace_size,
)
save_trt_engine(trt_engine, "yolox_x_pretrained.engine")