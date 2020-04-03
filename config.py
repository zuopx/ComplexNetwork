import torch


class GlobalVar:

    DB = 'E:/Projects/ComplexNetwork/db'

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_DB() -> str:
    return GlobalVar.DB


def get_DEVICE() -> torch.device:
    return GlobalVar.DEVICE
