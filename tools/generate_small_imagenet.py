import numpy as np
import os
import os.path as osp
import mmcv
import shutil
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(
        description='generate a smaller imagenet dataset from the original one'
    )
    parser.add_argument(
        'src_root',
        type=str,
        help='root of image folders of original imagenet')
    parser.add_argument(
        'dst_root',
        type=str,
        help='root for saving the new generated image folders')
    parser.add_argument(
        '--n',
        type=int,
        default=10,
        help='number of samples per class to keep')
    args = parser.parse_args()
    return args


def small_imagenet(src_root, dst_root, n=10):
    src_dirs = os.listdir(src_root)
    for src_dir in tqdm(src_dirs):
        imgs = os.listdir(osp.join(src_root, src_dir))
        mmcv.mkdir_or_exist(osp.join(dst_root, src_dir))
        imgs = np.random.choice(imgs, n, replace=False)
        for img in imgs:
            src_file = osp.join(src_root, src_dir, img)
            dst_file = osp.join(dst_root, src_dir, img)
            shutil.copy(src_file, dst_file)


def main():
    args = parse_args()
    small_imagenet(src_root=args.src_root, dst_root=args.dst_root, n=args.n)


if __name__ == "__main__":
    main()
