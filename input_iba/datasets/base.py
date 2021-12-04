from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self):
        pass

    @abstractmethod
    def get_ind_to_cls(self):
        raise NotImplementedError

    @abstractmethod
    def get_cls_to_ind(self):
        raise NotImplementedError
