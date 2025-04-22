import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from torchvision.transforms.functional import normalize


class ONNXRuntimeReIDInterface:
    def __init__(self, onnx_path, device='cuda', batch_size=8):
        self.batch_size = batch_size
        self.device = device
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])

        # Extract input/output names from ONNX model
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.cache = {}
        self.cache_name = None

    def preprocess(self, imgs):
        batch = list()
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256)).astype(np.float32) / 255.0
            img = np.moveaxis(img, -1, 0)  # HWC -> CHW
            img = torch.from_numpy(img)
            img = normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            batch.append(img)
        return batch

    def postprocess(self, features):
        # Normalize features (cosine distance)
        features = F.normalize(torch.from_numpy(features), dim=1)
        return features.numpy()

    def inference(self, image, detections):
        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)
        crops = []

        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
            crops.append(patch)

        features = np.zeros((0, 384))  # or 768, depending on your model

        # Batch-wise processing
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i+self.batch_size]
            batch_tensors = self.preprocess(batch)
            input_tensor = torch.stack(batch_tensors).numpy().astype(np.float32)

            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            normed = self.postprocess(outputs)
            features = np.vstack((features, normed))

        return features