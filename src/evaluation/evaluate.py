import os
import numpy as np
import trackeval
from src.utils import setup_logger

def evaluation(args, save_tracker_result_path, gt_folder):
    log_folder = setup_logger()
    tracker_folder = "/".join(save_tracker_result_path.split("/")[:2])
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'Identity', 'MAYI'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    config['GT_FOLDER'] = gt_folder
    config['TRACKERS_FOLDER'] = tracker_folder
    config['BENCHMARK'] = args.dataset
    config['SPLIT_TO_EVAL'] = args.tag

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    dataset_config['seqmap_tag'] = args.tag
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE, trackeval.metrics.MAYI]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    print(f"Finished Evaluating Benchmark: {args.dataset}, Split: {args.tag}\n")

    result_dict = {}
    for fname, res in output_res['MotChallenge2DBox']['last_benchmark'].items():
        HOTA = res['pedestrian']['HOTA']['HOTA'].mean() * 100
        IDF1 = res['pedestrian']['Identity']['IDF1'] * 100
        IDERROR = res['pedestrian']['MAYI']['ID_ERROR']
        FS = np.abs(res['pedestrian']['Count']['IDs'] - res['pedestrian']['Count']['GT_IDs'])/res['pedestrian']['Count']['GT_IDs'] * 100
        TIDP = res['pedestrian']['MAYI']['TIDP'] * 100
        TIDR = res['pedestrian']['MAYI']['TIDR'] * 100
        result_dict[fname] = [HOTA, IDF1, IDERROR, FS, TIDP, TIDR]

    # save csv file
    csv_path = os.path.join(log_folder, "result.csv")
    with open(csv_path, 'w') as f:
        f.write("detector,tracker,reid,appearance_thresh,MAX_CDIST_TH\n")
        f.write(f"{args.det_model},{args.tracker_model},{args.reid_model},{args.appearance_thresh},{args.MAX_CDIST_TH}\n")
        f.write("\n")

        f.write("video_name,HOTA,IDF1,IDERROR,FS,TIDP,TIDR\n")
        for fname, res in result_dict.items():
            f.write(f"{fname},{res[0]:.2f},{res[1]:.2f},{res[2]},{res[3]:.2f},{res[4]:.2f},{res[5]:.2f}\n")

    return [HOTA, IDF1, IDERROR, FS, TIDP, TIDR]