import torch


def using_gpu() -> bool:
    return get_device().type == "cuda"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
