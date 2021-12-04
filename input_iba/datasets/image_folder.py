import os
import os.path as osp

import cv2
from glob import glob

from .base import BaseDataset
from .builder import DATASETS, build_pipeline


@DATASETS.register_module()
class ImageFolder(BaseDataset):

    def __init__(self, img_root, pipeline, valid_formats=('png', )):
        assert isinstance(
            valid_formats,
            (list, tuple)), 'valid_formats must be either a list or tuple'
        super(ImageFolder, self).__init__()
        self.img_root = img_root

        cls_names = sorted(os.listdir(img_root))
        self.cls_to_ind = {c: i for i, c in enumerate(cls_names)}
        self.ind_to_cls = {v: k for k, v in self.cls_to_ind.items()}

        image_paths = []
        for valid_format in valid_formats:
            image_paths.extend(
                glob(
                    osp.join(self.img_root, f'**/*.{valid_format}'),
                    recursive=True))
        self.image_paths = image_paths
        self.pipeline = build_pipeline(pipeline)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_folder, img_name_with_ext = osp.split(img_path)
        img_name = osp.splitext(img_name_with_ext)[0]
        cls_name = osp.basename(img_folder)
        target = int(self.cls_to_ind[cls_name])

        res = self.pipeline(image=img)
        img = res['image']

        return dict(input=img, target=target, input_name=img_name)

    def __len__(self):
        return len(self.image_paths)

    def get_ind_to_cls(self):
        return self.ind_to_cls

    def get_cls_to_ind(self):
        return self.cls_to_ind
