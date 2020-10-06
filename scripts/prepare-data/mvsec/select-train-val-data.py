"""Use YOLOv3 to prepare train and val data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os
import glob
import shutil

import numpy as np

from mmdet.apis import init_detector, inference_detector

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true")
parser.add_argument("--val", action="store_true")

args = parser.parse_args()

# target classes
#  target_classes = ["bus", "car", "person", "stop_sign",
#                    "traffic_light", "truck"]
target_classes = ["car"]

# some helping paths
data_root = os.path.join(os.environ["HOME"], "data", "mvsec")

train_data = os.path.join(data_root, "train_data")
val_data = os.path.join(data_root, "val_data_3")

train_final_data = os.path.join(data_root, "train_data_final")
val_final_data = os.path.join(data_root, "val_data_3_final")

if args.train is True:
    source_dir = train_data
    target_dir = train_final_data
elif args.val is True:
    source_dir = val_data
    target_dir = val_final_data
    #  target_dir = val_data

# build ground truth folder
gt_folder = os.path.join(target_dir, "groundtruths")
if not os.path.isdir(gt_folder):
    os.makedirs(gt_folder)

# get the list of the file
file_list = sorted(glob.glob("{}".format(source_dir)+"/*.npz"))

# building model
config_file = os.path.join(
    os.environ["HOME"], "workspace", "mmdetection",
    "configs", "htc",
    "htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py")
checkpoint_file = os.path.join(
    os.environ["HOME"], "workspace", "mmdetection", "checkpoints",
    "htc_dconv_c3-c5_mstrain_400_1400_"
    "x101_64x4d_fpn_20e_20190408-0e50669c.pth")

# build model
model = init_detector(config_file, checkpoint_file, device="cuda:0")

print("Model built")

class_names = model.CLASSES
score_thr = 0.8

for file_path in file_list:
    print("Examining File: {}".format(file_path))
    file_data = np.load(file_path)
    # split the file name
    base_name = os.path.basename(file_path)[:-4]

    candidate_img = file_data["img"][..., np.newaxis]
    candidate_img = np.concatenate(
        (candidate_img, candidate_img, candidate_img), axis=2)
    #  candidate_img = file_data["img"]

    # detect image on the network
    detect_result = inference_detector(model, candidate_img)

    bboxes = np.vstack(detect_result[0])
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(detect_result[0])
    ]
    labels = np.concatenate(labels)

    # filter the final boxes
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    # if nothing detected, skip
    if len(bboxes) == 0 and len(labels) == 0:
        print("No boxes detected. Skipping {}".format(file_path))
        continue

    # check if there are boxes in the target class
    write_flag = False
    for box_id in range(bboxes.shape[0]):
        if model.CLASSES[labels[box_id]] in target_classes:
            write_flag = True
            break

    if write_flag is False:
        print("No writable boxes. Skipping {}".format(file_path))
        continue

    # if at least one object detected
    # copy the file to the final folder and write the ground truth text
    shutil.copyfile(file_path, os.path.join(target_dir, base_name+".npz"))

    # write ground truth
    gt_file = open(os.path.join(gt_folder, base_name+".txt"), "w+")
    for box_id in range(bboxes.shape[0]):
        if model.CLASSES[labels[box_id]] in target_classes:
            gt_file.write(
                "{} {} {} {} {}\n".format(
                    #  "tvmonitor",
                    model.CLASSES[labels[box_id]],
                    bboxes[box_id, 0],
                    bboxes[box_id, 1],
                    bboxes[box_id, 2],
                    bboxes[box_id, 3]))
    gt_file.close()

    print("Copied file {} and saved groundtruth".format(file_path))

print("Results dumped")
