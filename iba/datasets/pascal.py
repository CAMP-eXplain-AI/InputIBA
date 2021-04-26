from .builder import DATASETS, build_pipeline
from .base import BaseDataset
import os.path as osp
import cv2
from PIL import Image
import mmcv
import numpy as np
import torch
from albumentations.core.composition import BboxParams
from xml.etree import ElementTree as ET


@DATASETS.register_module()
class PascalVOC(BaseDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self,
                 img_root,
                 annot_root,
                 img_sets_file,
                 pipeline,
                 gt_mask_root=None,
                 with_mask=True,
                 force_one_hot=True,
                 preds_file=None,
                 min_size=None,
                 test_mode=False):
        super(PascalVOC, self).__init__()
        if with_mask:
            assert gt_mask_root is not None, f"If with_mask, gt_mask_root should not be None, but got {gt_mask_root}"
        if force_one_hot:
            assert preds_file is not None, f"If force_one_hot, pred_file should not be None, but got {preds_file}"

        self.img_root = img_root
        self.annot_root = annot_root
        self.cls_to_ind = {cls: i for i, cls in enumerate(self.CLASSES)}
        self.ind_to_cls = {i: cls for i, cls in enumerate(self.CLASSES)}

        bbox_params = BboxParams(format='pascal_voc', label_fields=['labels'])
        self.pipeline = build_pipeline(
            pipeline, default_args=dict(bbox_params=bbox_params))

        img_sets = mmcv.list_from_file(img_sets_file)
        self.with_mask = with_mask
        self.gt_mask_root = gt_mask_root
        self.img_sets = img_sets

        self.force_one_hot = force_one_hot
        if self.force_one_hot:
            self.preds = mmcv.load(preds_file)
        self.min_size = min_size
        self.test_mode = test_mode

    def get_cls_to_ind(self):
        return self.cls_to_ind

    def get_ind_to_cls(self):
        return self.ind_to_cls

    def __len__(self):
        return len(self.img_sets)

    def __getitem__(self, idx):
        img_name = self.img_sets[idx]
        img = cv2.cvtColor(
            cv2.imread(osp.join(self.img_root, f'{img_name}.jpg')),
            cv2.COLOR_BGR2RGB)
        bbox_info = self.get_bbox_info(img_name)
        bboxes = bbox_info['bboxes']
        labels = bbox_info['labels']
        if self.with_mask:
            masks = self.get_mask_info(img_name)
            transformed = self.pipeline(image=img,
                                        bboxes=bboxes,
                                        labels=labels,
                                        masks=masks)
        else:
            transformed = self.pipeline(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = np.array(transformed['bboxes']).astype(int)
        labels = np.array(transformed['labels']).astype(int)

        # filter bboxes and labels
        if self.force_one_hot:
            pred = np.array(self.preds[img_name]['pred'])
            if len(labels) > 0:
                # among the gt classes, only take the class which the classifier predicts with highest probability
                unique_labels = np.unique(labels)
                pred = pred[unique_labels]
                max_ind = np.argmax(pred)
                one_hot_cls = unique_labels[max_ind]
                bboxes = bboxes[labels == one_hot_cls]
                labels = labels[labels == one_hot_cls]
            else:
                # For background image, take the class with highest probability from all the classes
                one_hot_cls = np.argmax(pred)
                labels = [one_hot_cls]
        res = dict(input=img, bboxes=bboxes, input_name=img_name)

        if self.with_mask:
            masks = transformed['masks']
            masks = np.stack(masks, 0)
            res.update(masks=masks)

        # use integer as target
        target = labels[0]
        res.update(target=target)
        return res

    def get_bbox_info(self, img_name):
        xml_path = osp.join(self.annot_root, f'{img_name}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cls_to_ind[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(bboxes=bboxes.astype(np.float32),
                   labels=labels.astype(np.int64),
                   bboxes_ignore=bboxes_ignore.astype(np.float32),
                   labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_mask_info(self, img_name):
        gt_mask = np.array(
            Image.open(osp.join(self.gt_mask_root, f'{img_name}.png')))
        masks = []
        for label in range(len(self.CLASSES)):
            # class indices in the mask is in the interval [1, 20]
            masks.append((gt_mask == (label + 1)).astype(int))  # noqa
        if len(masks) == 0:
            masks = [np.zeros_like(gt_mask)]
        return masks
