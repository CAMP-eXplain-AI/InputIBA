from .base import BaseDataset
import os
import os.path as osp
from glob import glob
from .builder import DATASETS, build_pipeline
import cv2


@DATASETS.register_module()
class CIFAR10(BaseDataset):
    def __init__(self,
                 img_root,
                 pipeline):
        super(CIFAR10, self).__init__()
        self.img_root = img_root

        cls_names = sorted(os.listdir(img_root))
        self.cls_to_ind = {c:i for i, c in enumerate(cls_names)}
        self.ind_to_cls = {v: k for k, v in self.cls_to_ind.items()}

        self.image_paths = glob(osp.join(self.img_root, '**/*.png'), recursive=True)
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

        return dict(img=img,
                    target=target,
                    img_name=img_name)

    def __len__(self):
        return len(self.image_paths)

    def get_ind_to_cls(self):
        return self.ind_to_cls

    def get_cls_to_ind(self):
        return self.cls_to_ind