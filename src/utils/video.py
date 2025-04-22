
import boto3
import os
import json
import cv2
import numpy as np
import ffmpeg
import time
import subprocess
from tqdm import tqdm

# download video from s3 and extract frames and save them to the folder
class VideoExtractor:
    def __init__(self, bucket_name, video_key, base_folder="tmp"):
        self.bucket_name = bucket_name
        self.video_key = video_key
        self.s3 = boto3.client('s3')
        self.base_folder = base_folder
        self.local_video_path = os.path.join(self.base_folder, video_key)  # Maintain the directory structure
        self.location, date, cam, timestamp = video_key.split(".mp4")[0].split('/')
        self.local_frames_folder = os.path.join(self.base_folder, self.location, date, cam, timestamp)
        self.root_folder = os.path.join(self.base_folder, self.location, date, cam)

        self.video_length_seconds = 60

    def download_video(self, start_time):
        print(f"Downloading video from S3 bucket: {self.bucket_name}, key: {self.video_key}")
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(self.local_video_path), exist_ok=True)
        if not os.path.exists(self.local_video_path):
            self.s3.download_file(self.bucket_name, self.video_key, self.local_video_path)
        print(f"Video downloaded to {self.local_video_path}")
        self.trim_video(start_time=start_time)

    def trim_video(self, start_time=0):
        end_time = start_time + self.video_length_seconds
        start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time))
        end_time_str = time.strftime('%H:%M:%S', time.gmtime(end_time))
        print(f"Trimming video from {start_time_str} to {end_time_str}")
        
        # Ensure the output folder exists
        output_video_path = os.path.join(self.base_folder, self.video_key.split(".mp4")[0] + f"_{self.video_length_seconds}.mp4")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Escape paths to handle spaces and special characters
        input_path_escaped = f'"{self.local_video_path}"'
        output_path_escaped = f'"{output_video_path}"'

        # ffmpeg command
        if not os.path.exists(output_video_path):
            cmd = (
                f"ffmpeg -i {input_path_escaped} -ss {start_time_str} -to {end_time_str} "
                f"-c:v libx264 -preset fast -c:a aac {output_path_escaped}"
            )
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Check if ffmpeg succeeded
            if result.returncode != 0:
                print(f"Error during video trimming: {result.stderr.decode('utf-8')}")
                raise RuntimeError("Video trimming failed.")
        
        print(f"Video trimmed to {output_video_path}")
        self.local_video_path = output_video_path

    def extract_frames(self):
        if not os.path.exists(self.local_frames_folder):
            print(f"Extracting frames from video: {self.local_video_path}")
            # Ensure the parent directory exists
            os.makedirs(self.local_frames_folder, exist_ok=True)
            # Read the video
            cap = cv2.VideoCapture(self.local_video_path)
            frame_count = 0
            for i in tqdm(range(600)):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(self.local_frames_folder, f"{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            print(f"Extracted {frame_count} frames to {self.local_frames_folder}")
        else:
            print(f"Frames already extracted to {self.local_frames_folder}")


def extract_frames(video_path, output_dir, fps_target=10, max_frames=1500):
    os.makedirs(output_dir, exist_ok=True)

    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    fps = eval(video_info['r_frame_rate'])  # 비디오의 실제 FPS

    save_interval = max(1, round(fps / fps_target))

    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )

    frame_size = width * height * 3
    saved_count = 0
    for i in range(max_frames):
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break

        if i % save_interval == 0:
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(output_dir, f'{saved_count:06d}.jpg')
            cv2.imwrite(frame_path, frame_bgr)
            saved_count += 1

        if i % 100 == 0:  # 진행 상황 출력
            print(f"Processed {i}/{num_frames} frames, Saved {saved_count} frames")

    process.stdout.close()
    process.wait()
    print(f"Extracted {saved_count} frames at approximately {fps_target} FPS")