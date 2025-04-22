import onnx
from coram.tensorrt import onnx2trt, save_trt_engine

save_trt_path = "vit-small-ics_v2.engine"

onnx_model_path = "/maydrive/mayreid/onnx/vit_small_ics@mkldah_h256_d384.onnx"
onnx_model = onnx.load(onnx_model_path)

opt_shape_dict = {
    "input": [
        [1, 3, 256, 128],
        [8, 3, 256, 128],
        [16, 3, 256, 128],
    ],
}

max_workspace_size = 1 << 30
trt_engine = onnx2trt(
    onnx_model,
    opt_shape_dict,
    max_workspace_size=max_workspace_size,
)

save_trt_engine(trt_engine, save_trt_path)