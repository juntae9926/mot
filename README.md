# mot

Tracker 연구를 위한 라이브러리입니다.

## Dependencies

- Python ≥ 3.10
- CUDA ≥ 12.1
- PyTorch ≥ 2.1

## Install Environment

1. Clone the repository:

    ```bash
    git clone https://github.com/juntae9926/mot.git
    ```

2. Install TrackEval
    ```bash
    pip install -e ./TrackEval
    ```

6. Setup mot:

    ```bash
    pip install -e .
    ```

## Run Inference
```bash
# set detector, tracker, reid model
# use --help to set other parameters
python track_mot.py
python track_private.py
```
