import numpy as np
import cv2

def daram_result_filtering(results, polygon):
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