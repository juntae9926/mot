import os
import numpy as np
import json
import cv2
import shutil

DATA_PATH = '/mnt/hdd/tracker_benchmarks_public/MOT17'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['train_half', 'val_half', 'train', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'pedestrian'}]}

        # split 이름에 따라 하위 폴더명 설정
        if 'train_half' in split or split == 'train':
            split_folder = 'mot17-train'
        elif 'val_half' in split:
            split_folder = 'mot17-val'
        elif split == 'test':
            split_folder = 'mot17-test'
        else:
            split_folder = split

        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1

        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
                continue
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})

            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            seqinfo_src_path = os.path.join(seq_path, 'seqinfo.ini')

            images = os.listdir(img_path)
            num_images = len([img for img in images if 'jpg' in img])

            # 정확히 반으로 나누기
            if HALF_VIDEO and ('half' in split):
                image_range = [0, (num_images // 2) - 1] if 'train' in split else \
                              [num_images // 2, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            # seqinfo.ini 수정 저장
            if os.path.exists(seqinfo_src_path):
                with open(seqinfo_src_path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    if line.lower().startswith('seqlength'):
                        seq_length = image_range[1] - image_range[0] + 1
                        line = f"seqLength={seq_length}\n"
                    new_lines.append(line)
                seqinfo_dst_path = os.path.join(DATA_PATH, "gt", split_folder, seq, "seqinfo.ini")
                os.makedirs(os.path.dirname(seqinfo_dst_path), exist_ok=True)
                with open(seqinfo_dst_path, 'w') as f:
                    f.writelines(new_lines)

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(os.path.join(img_path, f'{i + 1:06d}.jpg'))
                height, width = img.shape[:2]
                image_info = {
                    'file_name': f'{seq}/img1/{i + 1:06d}.jpg',
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1 - image_range[0],
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height, 'width': width
                }
                out['images'].append(image_info)

            print(f'{seq}: {num_images} images')

            if split != 'test':
                det_path = os.path.join(seq_path, 'det/det.txt')
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')

                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([a for a in anns if image_range[0] <= int(a[0]) - 1 <= image_range[1]], dtype=np.float32)
                    anns_out[:, 0] -= image_range[0]

                    gt_eval_dir = os.path.join(DATA_PATH, "gt", split_folder, seq, "gt")
                    os.makedirs(gt_eval_dir, exist_ok=True)
                    gt_out = os.path.join(gt_eval_dir, "gt.txt")
                    with open(gt_out, 'w') as fout:
                        anns_out_sorted = anns_out[np.argsort(anns_out[:, 0])]
                        for o in anns_out_sorted:
                            fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                int(o[6]), int(o[7]), o[8]))

                if CREATE_SPLITTED_DET and ('half' in split):
                    dets_out = np.array([d for d in dets if image_range[0] <= int(d[0]) - 1 <= image_range[1]], dtype=np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, f'det/det_{split}.txt')
                    with open(det_out, 'w') as dout:
                        dets_out_sorted = dets_out[np.argsort(dets_out[:, 0])]
                        for o in dets_out_sorted:
                            dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                int(o[0]), int(o[1]), o[2], o[3], o[4], o[5], o[6]))

                print(f'{int(anns[:, 0].max())} ann images')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    if not ('15' in DATA_PATH):
                        if not int(anns[i][6]) == 1:
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:
                            category_id = -1
                        else:
                            category_id = 1
                            if track_id != tid_last:
                                tid_curr += 1
                                tid_last = track_id
                    else:
                        category_id = 1
                    ann_cnt += 1
                    ann = {
                        'id': ann_cnt,
                        'category_id': category_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]),
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)
            image_cnt += num_images
            print(tid_curr, tid_last)

        print(f'loaded {split} for {len(out["images"])} images and {len(out["annotations"])} samples')
        json.dump(out, open(out_path, 'w'))
