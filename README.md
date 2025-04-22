# maytracker

사내 Tracker 연구를 위한 라이브러리입니다.

## Dependencies

- Python ≥ 3.11
- CUDA ≥ 12.1
- PyTorch ≥ 2.1

## Dockerfile

To simplify installation, use the Dockerfile.

## Install Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/mAy-I/maytracker.git --recurse-submodules
    ```

2. Run the Docker container:

    ```bash
    # Update github_token
    ARG github_token="{GITHUB_TOKEN}"
    ```
    ```
    # Build and run the container
    docker build . -t maytracker
    docker run -v ~/maydrive:/maydrive -v ./:/workspace --ipc=host --gpus=all --name maytracker -it maytracker
    ```

3. Install Daram and Coram:

    ```bash
    # Install daram v3.18.3
    git clone -b v3.18.3 https://github.com/mAy-I/daram.git --recurse-submodules
    pip install -e ./daram

    # Locate TensorRT packages in /usr/local/cuda
    cp /usr/lib/x86_64-linux-gnu/libnvinfer* /usr/local/cuda/targets/x86_64-linux/lib

    # Install coram
    git clone https://github.com/mAy-I/coram.git
    export TENSORRT_DIR=/usr/local/cuda
    CORAM_WITH_OPS=1 CORAM_WITH_TRT=1 pip install -e ./coram
    ```

4. Install mayDetector
    ```bash
    git clone https://github.com/mAy-I/mayDetector.git
    pip install -e ./mayDetector
    ```

5. Install TrackEval
    ```bash
    pip install -e ./TrackEval
    ```

6. Setup maytracker:

    ```bash
    pip install -e .
    ```

## Create TensorRT Engine

### Detector

Use the script in the `weights/` folder to create the TensorRT engine for the detector.
```
cd ./weights
python detector_convert.py
```

### Tracker

Use the script in the `weights/` folder to create the TensorRT engine for the tracker.
```
cd ./weights
python reid_convert.py
```

## Run Inference
```bash
# set detector, tracker, reid model
# use --help to set other parameters
python main.py --det_model daram --tracker_model visiou --reid_model vit-small-ics --appearance_thresh 0.25 --MAX_CDIST_TH 0.155
```