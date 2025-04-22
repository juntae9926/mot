import os
import cv2
import json
import numpy as np
import torch


class PrivateDataset(torch.utils.data.Dataset):  # <- inherit Dataset
    def __init__(self, video_path, det_result_path, save_vis=False, vis_output_dir="results/vis_output"):
        self.video_path = video_path
        self.frame_folder = self._get_frame_folder(video_path)
        self.det_result = self._load_detections(det_result_path)
        self.frame_ids = sorted([int(k) for k in self.det_result.keys()])
        self.save_vis = save_vis
        self.vis_output_dir = vis_output_dir
        if save_vis and not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)

        if save_vis:
            self.visualize_detections()

    def _get_frame_folder(self, video_path):
        if os.path.isdir(video_path):
            return video_path
        elif os.path.isfile(video_path):
            folder_path = os.path.splitext(video_path)[0]
            assert os.path.isdir(folder_path), f"Frame folder not found: {folder_path}"
            return folder_path
        else:
            raise ValueError(f"Invalid video path: {video_path}")

    def _load_detections(self, det_result_path):
        with open(det_result_path, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        frame_name = f"{frame_id:06d}.jpg"
        img_path = os.path.join(self.frame_folder, frame_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        bboxes = []
        if str(frame_id) in self.det_result:
            for obj in self.det_result[str(frame_id)]["person_joints"]:
                t, l, b, r = obj["bbox"]
                score = obj.get("score", 1.0)
                cls = obj.get("class", 1)
                bboxes.append([t, l, b, r, score, cls])
        bboxes = np.array(bboxes) if bboxes else np.empty((0, 6))
        return frame_id, img, bboxes

    def visualize_detections(self):
        for idx in range(len(self)):
            img, bboxes, frame_id = self[idx]
            for bbox in bboxes:
                t, l, b, r, score, cls = bbox
                cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                cv2.putText(img, f"Score: {score:.2f}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            vis_path = os.path.join(self.vis_output_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(vis_path, img)
            print(f"Saved visualization for frame {frame_id} at {vis_path}")
