import os
import cv2
import json
import numpy as np

from src.utils.video import extract_frames
from src.utils.converter import (
    yolo_to_daram, 
    maydetector_to_daram, 
    mot_to_daram, 
    make_detection_from_tracking
)

from ultralytics import YOLO


class Detector:
    def __init__(self, det_model, weight_path=None, save_root='results'):
        self.save_root = save_root
        det_model = det_model.lower()
        if "ultralytics" in det_model:
            self.detector_type = "ultralytics"
            self.model = YOLO(weight_path)
            self.model.eval()

        elif det_model == "gt":
            self.detector_type = "gt"
            self.model = None
        else:
            raise ValueError(f"Unsupported detector type: {det_model}")

    def run(self, source, polygon=None):
        self.video_name = self._extract_video_name(source)
        os.makedirs(self.save_root, exist_ok=True)
        save_path = os.path.join(self.save_root, self.detector_type, f"{self.video_name}_detection.json")
        if os.path.exists(save_path):
            print(f"Detection results already exist for {source}. Skipping...")
            return
        else:
            print(f"Running detector on {source}...")
            os.makedirs(os.path.join(self.save_root, self.detector_type), exist_ok=True)

        if self.detector_type == "ultralytics":
            self._run_ultralytics(source, save_path, polygon)
        elif self.detector_type == "gt":
            self._run_gt(source, save_path)
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    def _extract_video_name(self, path):
        if os.path.isdir(path):
            return path.split('/')[-2]
        elif os.path.isfile(path):
            return os.path.splitext(os.path.basename(path))[0]
        else:
            raise ValueError(f"Invalid source path: {path}")

    def _ensure_frame_folder(self, source):
        """Ensure source is a folder of frames. If video, extract to folder."""
        if os.path.isdir(source):
            return source
        elif os.path.isfile(source) and source.endswith((".mp4", ".avi", ".mov", ".mkv")):
            frame_folder = os.path.splitext(source)[0]
            if not os.path.exists(frame_folder):
                print("Extracting frames...")
                extract_frames(source, frame_folder)
            return frame_folder
        else:
            raise ValueError("Unsupported source format.")

    def _run_ultralytics(self, source, save_path, polygon):
        frame_folder = self._ensure_frame_folder(source)
        print("Predicting with ultralytics...")
        results = self.model.predict(source=frame_folder, stream=True)

        daram_results = yolo_to_daram(results)
        if polygon:
            daram_results = self.daram_result_filtering(daram_results, polygon)

        # Save the resized results 
        with open(save_path, 'w') as f:
            json.dump(daram_results, f)


    def _run_gt(self, source, save_path):
        print("Converting GT to DARAM format...")
        _, daram_result_by_object = mot_to_daram(source)
        daram_det = make_detection_from_tracking(daram_result_by_object)
        with open(save_path, 'w') as f:
            json.dump(daram_det, f)

    def daram_result_filtering(self, results, polygon):
        polygon = [(y, x) for x, y in polygon]
        polygon = np.array(polygon, dtype=np.float32)
        filtered_results = {}
        for frame_id, frame_data in results.items():
            joints = frame_data['person_joints']
            for result in joints:
                t, l, b, r = result['bbox']
                center = [(l + r) / 2, (t + b) / 2]
                if cv2.pointPolygonTest(polygon, center, False) < 0:
                    if frame_id not in filtered_results:
                        filtered_results[frame_id] = {}
                        filtered_results[frame_id]['person_joints'] = []
                    filtered_results[frame_id]['person_joints'].append(result)
        return filtered_results