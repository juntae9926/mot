import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from multiprocessing import freeze_support
from src.models.detector import Detector
from src.evaluation import evaluation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trackers.factory import get_tracker
from src.data.mot import MOTDataset
from src.data.private import PrivateDataset


def parse_args():
    parser = argparse.ArgumentParser("tracker validation inference parser")
    parser.add_argument("--base_folder", type=str, default="/home/jtkim/maydrive/tracker_benchmarks_private", help="Base folder for video and frame storage")
    parser.add_argument("--result_folder", type=str, default="results", help="Base folder for video and frame storage")
    
    # dataset options
    parser.add_argument("--dataset", default="private", help="Dataset to evaluate")
    parser.add_argument("--from_nas", default=True, action='store_true', help="Download video from NAS")

    # detector options
    parser.add_argument("--det_model", type=str, default="ultralytics", choices=["ultralytics", "gt"], help="Detector model")
    parser.add_argument("--det_model_path", type=str, default="/home/jtkim/maydrive/mayDetector/weights/ultralytics/yolo11-x__(CrowdHuman__Cafe)__(CrowdHuman__Cafe__lg__samsung__handsome)/best.pt", help="Path to detector model")

    # tracker options
    parser.add_argument("--tracker_model", type=str, default="botsort", choices=["botsort", "bytetracker", "boosttrack", "cbiou", "sushi"])
    parser.add_argument("--reid_model", type=str, default="vit-small-ics", choices=["efficientnet", "vit-small-ics", "resnet50-ibn"])
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold for filtering")
    parser.add_argument("--min_box_area", type=float, default=10, help="Minimum box area for filtering")

    parser.add_argument("--MAX_CDIST_TH", type=float, default=0.155, help="set MAX_CDIST_TH for clustering")

    # evaluation options
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--tag', default="single", help='Split to evaluate')

    # BOTSORT, BYTETracker configs
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="High score threshold for tracking")
    parser.add_argument("--track_low_thresh", type=float, default=0.1, help="Low score threshold for tracking")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Matching threshold for association")
    parser.add_argument("--track_thresh", type=float, default=0.4, help="Threshold for tracking")
    parser.add_argument("--new_track_thresh", type=float, default=0.7, help="Threshold to consider a new track")
    parser.add_argument("--track_buffer", type=int, default=30, help="Track buffer (frames to keep lost tracks)")

    # ReID
    parser.add_argument("--with_reid", type=bool, default=True, help="Use ReID features")
    parser.add_argument("--reid_model_path", type=str, default="/home/jtkim/jt/new_maytracker/weights/vit-small-ics_v2.onnx")

    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_R50-ibn.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"weights/mot17_sbs_S50-ibn.pth", type=str, help="reid config file path")
    parser.add_argument("--proximity_thresh", type=float, default=0.5, help="Proximity threshold for IoU")
    parser.add_argument("--appearance_thresh", type=float, default=0.25, help="Appearance threshold for embedding distance")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run ReID model on (cuda or cpu)")

    # GMC (global motion compensation)
    parser.add_argument("--cmc_method", type=str, default="sparseOptFlow", help="Camera motion compensation method")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--ablation", action="store_true", help="Ablation flag for debugging or analysis")

    # MOT20 setting
    parser.add_argument("--mot20", action="store_true", help="Use MOT20 specific settings")

    return parser.parse_args()


def main(args):
    assert args.dataset == "private"
    
    base_path = f"{args.base_folder}/sources"
    seqmaps = f"{args.base_folder}/gt/seqmaps/private-{args.tag}.txt"
    gt_folder = f"{args.base_folder}/gt"
    with open(seqmaps, 'r') as f:
        video_names = [line.rstrip() for line in f][1:]
    videos = []
    for video_name in video_names:
        location, _data = video_name.split("__")
        cam = _data.split("_")[0]
        timestamp = _data.split("_")[1]
        date = datetime.utcfromtimestamp(int(timestamp) / 1000).strftime('%Y%m%d')
        video_length = _data.split("_")[2]
        video_path = f"{base_path}/{location}/{date}/{cam}/{timestamp}_{video_length}.mp4"
        videos.append(video_path)
    videos.sort()

    with open("src/data/polygons.json", "r") as f:
        polygons = json.load(f)
    polygons = {k: v for k, v in polygons.items() if k in video_names}
    
    detector = Detector(det_model=args.det_model, weight_path=args.det_model_path)

    for video in videos:
        tracker = get_tracker(args.tracker_model, args)

        location, date, cam, video_name = video.split("/")[-4:]
        timestamp, video_length = video_name.replace(".mp4", "").split("_")
        _key = f"{location}__{cam}"
        video_name_key = f"{_key}_{video_name}"
        detector.run(source=video, polygon=polygons.get(_key))
        det_json_path = os.path.join(detector.save_root, args.det_model, f"{timestamp}_{video_length}_detection.json")

        dataset = PrivateDataset(video, det_json_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        mot_result = []
        for frame_id, img, detections in tqdm(dataloader, desc=f"Tracking {video_name}"):
            tag = f"{video_name}:{frame_id}"
            outputs = tracker.update(detections, img, tag)
            for t in outputs:
                if t.is_activated:
                    tid = t.track_id
                    x1, y1, x2, y2 = map(int, t.tlbr)
                    w, h = x2 - x1, y2 - y1
                    mot_result.append([frame_id + 1, tid, x1, y1, w, h, -1, -1, -1])

        # Save mot results
        sequence_name = f"{video_name_key}"
        tracker_save_path = os.path.join(detector.save_root, f"{args.det_model}-{args.tracker_model}-reid{args.with_reid}", f"{args.dataset}-{args.tag}", "last_benchmark/data/" f"{sequence_name}.txt")
        os.makedirs(os.path.dirname(tracker_save_path), exist_ok=True)
        with open(tracker_save_path, "w") as f:
            for row in mot_result:
                f.write(",".join(map(str, row)) + "\n")

    # if args.eval:
    mean_values = evaluation(args, f"{args.result_folder}/{args.det_model}-{args.tracker_model}-reid{args.with_reid}/{args.dataset}-{args.tag}", gt_folder)

    metrics = ["HOTA", "IDF1", "ID_ERROR", "LD", "FS", "TIDP", "TIDR"]
    values = [
        f"{mean_values[0]:.2f}",
        f"{mean_values[1]:.2f}",
        f"{mean_values[2]:.2f}",
        f"{mean_values[3]:.2f}",
        f"{mean_values[4]:.2f}",
        f"{mean_values[5]:.2f}",
    ]
    table = [metrics, values]
    table_str = tabulate(table, headers="firstrow", tablefmt="grid")
    logger.info("\n" + table_str)
                    

if __name__ == "__main__":
    freeze_support()
    args = parse_args()

    main(args)