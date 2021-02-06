from xml.etree import ElementTree as ET
import numpy as np


def load_voc_bboxes(xml_file, name_to_ind_dict, ignore_difficult=False):
    """Load bounding box annotations from an xml file.

    Args:
        xml_file (str): xml file path.
        name_to_ind_dict (dict): a dict mapping class names to integers.
        ignore_difficult (bool, optional): if True, ignore the difficult bounding boxes. Otherwise, concatenate the
            difficult boxes with the other boxes.

    Returns:
        (dict): contains two fields:
            bboxes: ndarray, bounding boxes with shape (num_bboxes, 4).
            labels: ndarray, labels with shape (num_bboxes,).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = name_to_ind_dict[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        # Coordinates may be float type
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        if difficult:
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
        labels = np.array(labels, dtype=int)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,), dtype=int)
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore, dtype=int)
    if not ignore_difficult:
        bboxes = np.concatenate([bboxes, bboxes_ignore], axis=0)
        labels = np.concatenate([labels, labels_ignore], axis=0)
    return dict(bboxes=bboxes, labels=labels)