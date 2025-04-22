import os

def sort_key(x):
    x_list = x.split(",")
    return (int(x_list[0]), int(x_list[1]))


def maydetector_to_daram(yolo_results):
    daram_data = {}
    for frame_data in yolo_results:
        frame_idx = str(int(str(frame_data['img_file']).split('.jpg')[0].split('/')[-1]))
        daram_data[frame_idx] = {}
        daram_data[frame_idx]["person_joints"] = []
        for obj in range(frame_data['boxes']['xyxy'].shape[0]):
            conf = frame_data['conf'][obj].item()
            xyxy_bbox = [int(v) for v in frame_data['boxes']['xyxy'][obj].tolist()]
            joints = [0 for _ in range(34)]
            daram_data[frame_idx]["person_joints"].append({
                "bbox": xyxy_bbox,
                "score": conf,
                "joints": joints
            })
    return daram_data


def yolo_to_daram(yolo_results):
    daram_data = {}
    for idx, frame_data in enumerate(yolo_results):

        frame_idx = str(idx)
        daram_data[frame_idx] = {}
        daram_data[frame_idx]["person_joints"] = []
        for obj in range(frame_data.boxes.shape[0]):
            conf = frame_data.boxes[obj].conf.item()
            xyxy_bbox = frame_data.boxes[obj].xyxy[0].tolist()
            xyxy_bbox = [int(x) for x in xyxy_bbox]
            joints = make_joints_from_bbox(xyxy_bbox)
            daram_data[frame_idx]["person_joints"].append({
                "bbox": xyxy_bbox,
                "score": conf,
                "joints": joints
            })
    return daram_data


def mot_to_daram(mot_data_path, result_fps=5):
    with open(mot_data_path, 'r') as f:
        mot_data = f.readlines()

    tracking_result_by_frame = {}
    for line in mot_data:
        frame_id, tid, x, y, w, h, _, _, _ = line.strip().split(",")
        frame_id = int(float(frame_id)) - 1  # Convert to integer and zero-indexed
        
        # result_fps가 5이면 홀수 frame_id는 제외
        if result_fps == 5 and frame_id % 2 == 1:
            continue
        
        tid = int(float(tid))
        x, y, w, h = map(lambda x: int(float(x)), [x, y, w, h])

        frame_id_str = str(frame_id)
        if frame_id_str not in tracking_result_by_frame:
            tracking_result_by_frame[frame_id_str] = {"person_joints": {}}
        
        if tid not in tracking_result_by_frame[frame_id_str]["person_joints"]:
            tracking_result_by_frame[frame_id_str]["person_joints"][str(tid)] = {}
        
        joints = make_joints_from_bbox([x, y, x+w, y+h])
        tracking_result_by_frame[frame_id_str]["person_joints"][str(tid)] = {
            "bbox": [[x, y], [x+w, y+h]],
            "score": 1.0,
            "joints": joints
        }

    tracking_result_by_object = {}
    for frame_id_str, frame_data in tracking_result_by_frame.items():
        for tid_str, joint_data in frame_data['person_joints'].items():
            if tid_str not in tracking_result_by_object:
                tracking_result_by_object[tid_str] = {"person_joints": {}}
            tracking_result_by_object[tid_str]["person_joints"][frame_id_str] = joint_data

    return tracking_result_by_frame, tracking_result_by_object


def daram_to_mot(daram_data):
    mot_results = []
    for fid, v in daram_data.items():
        bbox_dict = v['person_joints']
        for tid, res in bbox_dict.items():
            conf = res['score']
            bbox = res['bbox']
            x, y = bbox[0]
            w = bbox[1][0] - bbox[0][0]
            h = bbox[1][1] - bbox[0][1]
            mot_results.append(
                f"{int(fid)+1},{tid},{x},{y},{w},{h},{conf:.2f},-1,-1\n"
            )

    mot_results.sort(key=sort_key) 
    
    return mot_results


def make_joints_from_bbox(bbox):
    left_top_point = [int(bbox[0]), int(bbox[1])]
    right_top_point = [int(bbox[2]), int(bbox[1])]
    left_bottom_point = [int(bbox[0]), int(bbox[3])]
    right_bottom_point = [int(bbox[2]), int(bbox[3])]
    left_foot_point = [
        int(bbox[0]),
        int(bbox[1] + (bbox[3] - bbox[1]) * 0.8),
    ]
    right_foot_point = [
        int(bbox[2]),
        int(bbox[1] + (bbox[3] - bbox[1]) * 0.8),
    ]
    joints = (
        [0] * 10
        + left_top_point
        + right_top_point
        + [0] * 8
        + left_bottom_point
        + right_bottom_point
        + [0] * 4
        + left_foot_point
        + right_foot_point
    )
    return joints


def make_detection_from_tracking(tracking_result_by_object):
    detection_result = {}
    for tid_str, joint_data in tracking_result_by_object.items():
        for frame_id_str, frame_data in joint_data['person_joints'].items():
            if frame_id_str not in detection_result:
                detection_result[frame_id_str] = {"person_joints": []}

            x1, y1 = frame_data['bbox'][0]
            x2, y2 = frame_data['bbox'][1]
            frame_data['bbox'] = [x1, y1, x2, y2]

            detection_result[frame_id_str]["person_joints"].append(frame_data)
            
    return detection_result