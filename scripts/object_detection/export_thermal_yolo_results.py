"""Export event-driven YOLO result.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import glob
import os
import argparse
import json

import numpy as np
import torch
import torchvision.transforms as transforms

from evtrans.yolo.utils.utils import non_max_suppression
from evtrans.yolo.utils.utils import rescale_boxes
from evtrans.yolo.utils.utils import load_classes
from evtrans.yolo.utils.datasets import pad_to_square
from evtrans.yolo.utils.datasets import resize
from evtrans.pyyolo.model import YoloNetV3
from evtrans.networks import EVYOLOFrontend
from evtrans.networks import YOLODetectNet
from evtrans.networks import FrameYOLOFrontend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#  get on a parser
parser = argparse.ArgumentParser()

# image size
parser.add_argument("--img_size", type=int, default=416)
# validation folder
parser.add_argument("--val_data_dir", type=str)
# logging directory
parser.add_argument("--checkpoint", type=str)
# detection saving path
parser.add_argument("--detection_path", type=str)
parser.add_argument("--output_save_path", type=str, default="")

# number of convolution input dimension
parser.add_argument("--conv_input_dim", type=int)

# network weight path
parser.add_argument("--weights_path", type=str,
                    default=os.path.join("..", "weights",
                                         "yolov3_original.pth"))
# cutoff layer
parser.add_argument("--cut_stage", type=int)

parser.add_argument("--vis", action="store_true")

args = parser.parse_args()

# create detection path
if not os.path.isdir(args.detection_path):
    os.makedirs(args.detection_path)
    print("Detection path {} is created.".format(args.detection_path))

# load pretrained model
pretrain_model = YoloNetV3(nms=True)
pretrain_model.load_state_dict(torch.load(args.weights_path))
pretrain_model.to(DEVICE)
pretrain_model.eval()

# check checkpoint
if os.path.isfile(args.checkpoint):
    checkpoint_path = args.checkpoint
elif os.path.isdir(args.checkpoint):
    # search for the best checkpoint
    log_path = os.path.join(args.checkpoint, "log.csv")
    exp_log = np.loadtxt(log_path, delimiter=",", skiprows=1,
                         dtype=float)
    val_log = exp_log[:, 8]
    checkpoint_idx = np.argmin(val_log)+1
    checkpoint_path = os.path.join(
        args.checkpoint, "checkpoints",
        "checkpoint_{}.pt".format(int(checkpoint_idx)))
    print("Using checkpoint {}".format(checkpoint_path))

checkpoint = torch.load(checkpoint_path)
print("Checkpoint loaded.")

# build Event driven front end
ev_yolo = EVYOLOFrontend(stage=args.cut_stage,
                         input_dim=args.conv_input_dim).to(DEVICE)
ev_yolo.load_state_dict(checkpoint["model_state_dict"])
ev_yolo.eval()
print("Parameter loaded.")

if args.vis is True:
    yolo_frontend = FrameYOLOFrontend(
        pretrain_model.darknet, stage=args.cut_stage).to(DEVICE)
    yolo_frontend.eval()

yolo_detection_net = YOLODetectNet(
    pretrain_model, cut_stage=args.cut_stage, nms=True)
yolo_detection_net.eval()

files_list = sorted(glob.glob("{}".format(args.val_data_dir)+"/*.npz"))

classes = load_classes("../configs/coco.names")

with open("../configs/ilsvrc.names") as f:
    name_map = json.load(f)
    f.close()

print("Prepare to generate detections")

# inference the images
for file_idx in range(len(files_list)):
    # read the data
    file_path = files_list[file_idx]
    base_name = os.path.basename(file_path)[:-4]

    data = np.load(file_path)
    img = transforms.ToTensor()(data["ev_img"])

    if args.vis is True:
        ori_img = transforms.ToTensor()(data["img"])

    # get image shape
    img_shape = img.shape[1:3]

    img, _ = pad_to_square(img, 0)
    img = resize(img, args.img_size)

    if args.vis is True:
        ori_img, _ = pad_to_square(ori_img, 0)
        ori_img = resize(ori_img, args.img_size)

    candidate_ev_vol = img

    # inference on the data
    with torch.no_grad():
        candidate = candidate_ev_vol.unsqueeze(0).to(DEVICE)
        pre_detections = ev_yolo(candidate)

        detections = yolo_detection_net(pre_detections)

        detections = non_max_suppression(
            detections, 0.6, 0.4)

    if len(detections) == 0 or detections[0] is None:
        print("No box for {}".format(file_path))
        continue

    if args.vis is False:
        detections = rescale_boxes(detections[0], args.img_size, img_shape)
        detections = detections.cpu().data.numpy()
    else:
        detections = detections[0].cpu().data.numpy()

    # write ground truth
    detect_path = os.path.join(
        args.detection_path, base_name+".txt")
    detect_file = open(detect_path, "w+")
    for box_id in range(detections.shape[0]):
        try:
            detect_file.write(
                "{} {} {} {} {} {}\n".format(
                    classes[
                        int(detections[box_id, 6])].replace(" ", "_"),
                    detections[box_id, 4],
                    detections[box_id, 0],
                    detections[box_id, 1],
                    detections[box_id, 2],
                    detections[box_id, 3]))
        except KeyError:
            continue
    detect_file.close()

    if args.vis is True:
        with torch.no_grad():
            #  candidate = torch.cat([candidate, candidate, candidate], dim=1)
            candidate = ori_img.unsqueeze(0).to(DEVICE)
            frame_pre_detection = yolo_frontend(candidate)
            detections = yolo_detection_net(frame_pre_detection)

            detections = non_max_suppression(
                detections, 0.8, 0.4)

            if len(detections) == 0 or detections[0] is None:
                print("No APS box for {}".format(file_path))
                detections = np.zeros((0, 1))
            else:
                detections = detections[0].cpu().data.numpy()

    if args.vis is True:
        if not os.path.isfile(detect_path):
            continue

        dt = np.loadtxt(detect_path, usecols=(2, 3, 4, 5))
        if len(dt.shape) == 1:
            dt = [dt]

        # show your face
        f = plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(221)
        plt.imshow(ori_img.permute(1, 2, 0))
        plt.title("Intensity Frame")
        for box_id in range(detections.shape[0]):
            try:
                rect = patches.Rectangle(
                    (detections[box_id, 0],
                     detections[box_id, 1]),
                    (detections[box_id, 2]-detections[box_id, 0]),
                    (detections[box_id, 3]-detections[box_id, 1]),
                    linewidth=1, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)
            except KeyError:
                pass

        ax1 = plt.subplot(222)
        ev_vol_data = candidate_ev_vol.cpu().data.numpy()
        ev_vol_data = ev_vol_data.transpose(1, 2, 0)
        ev_vol_data = np.concatenate(
            (ev_vol_data, ev_vol_data, ev_vol_data),
            axis=2)
        plt.imshow(ev_vol_data, cmap="gray")
        plt.title("Thermal Image")
        for box in dt:
            box = box.astype(int)
            box[box < 0] = 0
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='blue', facecolor='none')
            ax1.add_patch(rect)
        for box_id in range(detections.shape[0]):
            try:
                rect = patches.Rectangle(
                    (detections[box_id, 0],
                     detections[box_id, 1]),
                    (detections[box_id, 2]-detections[box_id, 0]),
                    (detections[box_id, 3]-detections[box_id, 1]),
                    linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)
            except KeyError:
                pass

        plt.subplot(223)
        pre_detect_data = frame_pre_detection.cpu().data.numpy()
        sns.heatmap(pre_detect_data[0].mean(axis=0),
                    cmap="gray", square=True)
        plt.title("Frame Frontend Representation")

        plt.subplot(224)
        pre_detect_data = pre_detections.cpu().data.numpy()
        sns.heatmap(pre_detect_data[0].mean(axis=0),
                    cmap="gray", square=True)
        plt.title("Thermal Frontend Representation")
        plt.tight_layout()

        plt.show()

        f.clear()
        plt.close()

    print("Generated boxes for {}".format(file_path))

print("Results dumped")
