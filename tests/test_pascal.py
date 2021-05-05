from iba.datasets import build_dataset

import mmcv


def test_pascal():
    cfg = mmcv.Config.fromfile('configs/vgg_pascal.py')
    train_set = build_dataset(cfg.data['train'])
    _ = train_set[0]


if __name__ == '__main__':
    test_pascal()
