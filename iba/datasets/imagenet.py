from .builder import DATASETS
from .base import BaseDataset
from .pipelines.compose import Compose
import os.path as osp
from glob import glob
import json
from PIL import Image


@DATASETS.register_module()
class ImageNet(BaseDataset):
    """ImageNet. The folders should be structured as follows:
        root/
            class_folder_1/xxx.JPEG
            class_folder_1/yyy.JPEG
            ...
            class_folder_2/zzz.JPEG
            ...

    Args:
        root (str): root of the dataset.
        ind_to_cls_file(str): json file that contains mapping from indices to class names and sub-folder names.
        pipeline (list): pipeline to transform the images.
    """
    def __init__(self, root, ind_to_cls_file, pipeline):
        super(ImageNet, self).__init__()
        self.root = root

        with open(ind_to_cls_file, 'r') as f:
            ind_to_cls = json.load(f)
        self.dir_to_ind = {v[0]: int(k) for k, v in ind_to_cls.items()}
        self.ind_to_cls = {int(k): v[1] for k, v in ind_to_cls.items()}
        self.cls_to_ind = {v: k for k, v in self.ind_to_cls.items()}

        self.pipeline = Compose(pipeline)

        self.image_paths = glob(osp.join(root, '**/*.JPEG'), recursive=True)

    def __getitem__(self, index):
        """Get a single sample.

        Args:
            index (int): index of sample.

        Returns:
            A tuple of:
                img (Tensor): image tensor with shape (3, H, W).
                target (int): class index.
                img_name (int): base name of the image file.
        """
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = self.pipeline(img)

        dir_name = osp.basename(osp.dirname(img_path))
        img_name = osp.splitext(osp.basename(img_path))[0]
        target = int(self.dir_to_ind[dir_name])
        return img, target, img_name

    def __len__(self):
        return len(self.image_paths)

    def get_ind_to_cls(self):
        """Get a dict mapping class indices to class names"""
        return self.ind_to_cls

    def get_cls_to_ind(self):
        """Get a dict mapping class names to class indices"""
        return self.cls_to_ind