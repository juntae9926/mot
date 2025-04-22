import os
import json
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class MOTDataset(Dataset):
    def __init__(self, data_dir, video_name, json_file="val_half.json", det_json_file=None, save_vis=False, vis_output_dir="results/vis_output"):
        self.data_dir = data_dir
        self.video_name = video_name

        total_ids = sorted([int(img_id.split('.')[0]) for img_id in os.listdir(os.path.join(data_dir, "train", video_name, 'img1')) if img_id.endswith('.jpg')])
        self.frame_length = len(total_ids)
        if 'train' in json_file:
            self.ids = total_ids[:self.frame_length//2]
            self.type = 'train'
        elif 'val' in json_file:
            self.ids = total_ids[self.frame_length//2:]
            self.type = 'val'

        self.det_result = self._load_det_json(det_json_file) if det_json_file else None

        self.data = self._load_data()

        self.vis_output_dir = vis_output_dir
        if save_vis and not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)

        if save_vis:
            self.visualize_detections()


    def _load_det_json(self, det_json_file):
        with open(det_json_file, "r") as f:
            return json.load(f)

    def _load_data(self):
        data = []
        for frame_id in self.ids:
            img_path = os.path.join(self.data_dir, "train", self.video_name, "img1", f"{frame_id:06d}.jpg")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")

            if self.det_result and str(frame_id-1) in self.det_result:
                bboxes = []
                for obj in self.det_result[str(frame_id-1)]["person_joints"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    score = obj.get("score", 1.0)
                    cls = obj.get("class", 1)
                    bboxes.append([x1, y1, x2, y2, score, cls])
                bboxes = np.array(bboxes) if bboxes else np.empty((0, 6))
            else:
                NotImplementedError("Detection results not provided for this dataset.")
                """
                implementation for public detection
                """
            
            data.append((frame_id, img_path, bboxes))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_id, img_path, bboxes = self.data[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        if self.type == 'val':
            frame_id = int(frame_id) - self.frame_length // 2
        
        return frame_id, img, bboxes

    def visualize_detections(self):
        for idx in range(len(self)):
            frame_id, img, bboxes = self[idx]
            for bbox in bboxes:
                x1, y1, x2, y2, score, cls = bbox
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"Score: {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            vis_path = os.path.join(self.vis_output_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(vis_path, img)
            print(f"Saved visualization for frame {frame_id} at {vis_path}")
