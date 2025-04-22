import os
import sys
import json
import argparse
from loguru import logger
from datetime import datetime
from tabulate import tabulate
from multiprocessing import freeze_support
from src.configs import get_daram_detector_config, get_daram_tracker_config
from src.models import run_yolo_detector, run_daram_detector, run_maydetector, run_daram_tracker, run_gtdetector
from src.utils import daram_to_mot
from src.evaluation import evaluation

import daram
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser("tracker validation inference parser")
    parser.add_argument("--base_folder", type=str, default="/maydrive/tracker_benchmarks_private", help="Base folder for video and frame storage")
    parser.add_argument("--result_folder", type=str, default="engine_results", help="Base folder for video and frame storage")
    
    # dataset options
    parser.add_argument("--dataset", default="private", help="Dataset to evaluate")
    parser.add_argument("--from_nas", default=True, action='store_true', help="Download video from NAS")

    # detector options
    parser.add_argument("--det_model", type=str, default="sqzb", choices=["sqzb", "maydetector", "daram", "ultralytics", "gt"], help="Detector model")
    parser.add_argument("--det_model_path", type=str, default="/maydrive/mayDetector/weights/yolo11-x__coco__CrowdHuman__mar17/weights/best.pt", help="Path to detector model")

    # tracker options
    parser.add_argument("--tracker_model", type=str, default="visiou", choices=["vispose", "visiou", "litevisiou"])
    parser.add_argument("--reid_model", type=str, default="vit-small-ics", choices=["efficientnet", "vit-small-ics"])
    parser.add_argument("--appearance_thresh", type=float, default=0.25, help="set appearance_thresh")
    parser.add_argument("--MAX_CDIST_TH", type=float, default=0.05, help="set MAX_CDIST_TH for clustering")

    # evaluation options
    parser.add_argument('--eval', default=True, action='store_true', help='Run evaluation')
    parser.add_argument('--tag', default="every", help='Split to evaluate')

    return parser.parse_args()


def main(args):
    with open("polygons.json", 'r') as f:
        polygons = json.load(f)

    if args.dataset == "private":
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
        print(videos)
    
    for video_path in videos:
        # prepare video
        location, date, cam, video_name = video_path.split("/")[-4:]
        timestamp, video_length = video_name.split("_")[0], video_name.split("_")[1].split(".")[0]
        root_folder = os.path.join(args.result_folder, location, date, cam)
        os.makedirs(root_folder, exist_ok=True)
        video = daram.Video(video_path)
        _key = f"{location}__{cam}"

        # run detector
        if "ultralytics" in args.det_model:
            det_save_path = os.path.join(root_folder, "detection_result_yolo11-X.json")
            run_yolo_detector(args.det_model_path, video_path, det_save_path, polygons[_key])
        elif args.det_model == "daram":
            daram_detector_config = get_daram_detector_config()
            det_save_path = os.path.join(root_folder, "detection_result_yolox.json")
            run_daram_detector(daram_detector_config, video, det_save_path, polygons[_key])
        elif args.det_model == "sqzb":
            daram_detector_config = get_daram_detector_config()
            daram_detector_config.model_path = "weights/yolox_x_sqzb.engine"
            det_save_path = os.path.join(root_folder, "detection_result_yolox.json")
            run_daram_detector(daram_detector_config, video_path, det_save_path, polygons[_key])
        elif args.det_model == "maydetector":
            det_save_path = os.path.join(root_folder, "detection_result_maydetector.json")
            run_maydetector(args.det_model_path, video_path, det_save_path, polygons[_key]) 
        else:
            det_save_path = os.path.join(root_folder, "detection_result_gt.json")
            gt_path = os.path.join(args.base_folder, "gt", f"{args.dataset}-{args.tag}", f"{location}__{cam}_{timestamp}_{video_length}/gt/gt.txt")
            run_gtdetector(gt_path, det_save_path)
        video.data.add_json("detection_result", det_save_path)

        # run daram tracker
        daram_tracker_config = get_daram_tracker_config(args)
        run_daram_tracker(args, daram_tracker_config, video, root_folder)
        with open(f"{root_folder}/tracking_result_by_frame.json") as f:
            daram_data = json.load(f)

        # convert daram to mot format
        mot_results = daram_to_mot(daram_data)
        mot_save_path = f"{args.result_folder}/{args.tracker_model}/{args.dataset}-{args.tag}/last_benchmark/data/" + f"{location}__{cam}_{timestamp}_{video_length}.txt"
        os.makedirs(os.path.dirname(mot_save_path), exist_ok=True)
        with open(mot_save_path, 'w') as f:
            f.writelines(mot_results)

    if args.eval:
        mean_values = evaluation(args, f"{args.result_folder}/{args.tracker_model}/{args.dataset}-{args.tag}", gt_folder)
        metrics = ["HOTA", "IDF1", "ID_ERROR", "FS", "TIDP", "TIDR"]
        values = [
            f"{mean_values[0]:.2f}",
            f"{mean_values[1]:.2f}",
            f"{mean_values[2]:.2f}",
            f"{mean_values[3]:.2f}",
            f"{mean_values[4]:.2f}",
            f"{mean_values[5]:.2f}"
        ]
        table = [metrics, values]
        table_str = tabulate(table, headers="firstrow", tablefmt="grid")
        logger.info("\n" + table_str)
                    

if __name__ == "__main__":
    freeze_support()
    args = parse_args()
    main(args)