import os
import torch

from src.data.mot import MOTDataset
from src.data.private import PrivateDataset
from .transform import ValTransform

def get_mot_loader(dataset, test, data_dir="data", workers=4, size=(800, 1440)):
    # Different dataset paths
    if dataset == "mot17":
        direc = "MOT17"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    elif dataset == "mot20":
        direc = "MOT20"
        if test:
            name = "test"
            annotation = "test.json"
        else:
            name = "train"
            annotation = "val_half.json"
    else:
        raise RuntimeError("Specify path here.")

    # Same validation loader for all MOT style datasets
    valdataset = MOTDataset(
        data_dir=os.path.join(data_dir, direc),
        json_file=annotation,
        img_size=size,
        name=name,
        preproc=ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        ## preproc=ValTransform(rgb_means=(0.0, 0.0, 0.0), std=(1.0, 1, 1.0),)
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)
    dataloader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = 1
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    return val_loader



def get_detection_loader(dataset="private", test=False, data_dir="data", workers=4, size=(608, 1088)):
    """
    Load detection dataset in the same interface as get_mot_loader.
    """

    if dataset == "private":
        video_folder = os.path.join(data_dir, "private", "frames")
        annotation_file = os.path.join(data_dir, "private", "annotations", "det_result.json")
    else:
        raise RuntimeError("Specify a supported dataset like 'private'.")

    valdataset = PrivateDataset(
        video_path=video_folder,
        det_result_path=annotation_file,
        img_size=size,
        preproc=None,  # 필요 시 ValTransform 등 사용
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)
    dataloader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "sampler": sampler,
        "batch_size": 1,
    }
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
    return val_loader