############################################################
# This Python script generates folder structure for torchvision.datasets.DatasetFolder
# Each subfolder contains all images of same class (label)
# usage python reorder_imagenet.py -p $imagenet_path -gt $ground_truth
############################################################

import os
import json
import argparse
import random
from shutil import copyfile

def argparser():
    parser = argparse.ArgumentParser(description='read imagenet path')
    parser.add_argument('-p', '--imagenet_path', type=str, default="/home/yang/下载/ILSVRC2012_img_val",
                        help='path of ImageNet folder')
    parser.add_argument('-np', '--new_path', type=str, default="/home/yang/下载/train",
                        help='path of ImageNet folder')
    parser.add_argument('-n', '--number', type=int, default=10,
                        help='number of samples per class')
    return parser

def small_imagenet(args):
    # open class index dict
    with open("IBA/notebooks/imagenet_class_index.json") as index_json:
        index_dict = json.load(index_json)

    # create subfolders
    new_imagenet_path = args.new_path
    try:
        os.mkdir(new_imagenet_path)
    except OSError:
        print("Creation of the directory {} failed, directory may already exist!".format(new_imagenet_path))
    for folder_name in index_dict.values():
        try:
            os.mkdir(os.path.join(new_imagenet_path, folder_name[0]))
        except OSError:
            print("Creation of the directory {} failed, directory may already exist!".format(folder_name[0]))

    # move a small batch of images into new folder
    imagenet_path = args.imagenet_path
    for file in os.listdir(imagenet_path):
        if os.path.isdir(os.path.join(imagenet_path, file)):
            class_folder = file
            image_files = os.listdir(os.path.join(imagenet_path, file))
            sampled_image_files = random.sample(image_files, args.number)
            for sample in sampled_image_files:
                copyfile(os.path.join(imagenet_path, class_folder, sample),
                         os.path.join(new_imagenet_path, class_folder, sample))

if __name__ == "__main__":
    print("Start generate small imagenet")
    parser = argparser()
    args = parser.parse_args()
    small_imagenet(args)
    print("Done!")