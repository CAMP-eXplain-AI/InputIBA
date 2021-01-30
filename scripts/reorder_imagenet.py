############################################################
# This Python script generates folder structure for torchvision.datasets.DatasetFolder
# Each subfolder contains all images of same class (label)
# usage python reorder_imagenet.py -p $imagenet_path -gt $ground_truth
############################################################

import os
import json
import argparse

def argparser():
    parser = argparse.ArgumentParser(description='read imagenet path')
    parser.add_argument('-p', '--imagenet_path', type=str, default="/home/yang/下载/ILSVRC2012_img_val",
                        help='path of ImageNet folder')
    parser.add_argument('-gt', '--ground_truth', type=str, default="/home/yang/下载/val.txt",
                        help='path of ImageNet folder')
    return parser

def reorder_imagenet(args):
    # open class index dict
    with open("iba/notebooks/imagenet_class_index.json") as index_json:
        index_dict = json.load(index_json)

    # create subfolder
    imagenet_path = args.imagenet_path
    for folder_name in index_dict.values():
        try:
            os.mkdir(os.path.join(imagenet_path, folder_name[0]))
        except OSError:
            print("Creation of the directory {} failed, directory may already exist!".format(folder_name[0]))

    # load ground_truth
    with open(args.ground_truth) as f:
        ground_truth = f.readlines()
        ground_truth = [i.split("\n")[0].split(" ")[1] for i in ground_truth]

    # move images into belonging folder
    for file in os.listdir(imagenet_path):
        if os.path.isfile(os.path.join(imagenet_path, file)):
            file_name = file.split(".")[0]
            index = int(file_name.split("_")[-1])
            label = ground_truth[index-1]
            class_folder = index_dict[label][0]
            os.rename(os.path.join(imagenet_path, file), os.path.join(imagenet_path, class_folder, file))

if __name__ == "__main__":
    print("Start reorder imgs")
    parser = argparser()
    args = parser.parse_args()
    reorder_imagenet(args)
    print("Done!")