"""Show prediction.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# image path
img_number = "06623"
data_path = os.path.join(os.environ["HOME"], "data", "mvsec",
                         "val_data_final")
img_path = os.path.join(data_path, "frame_ev_pair_{}.npz".format(img_number))
gt_path = os.path.join(data_path, "pretrained_yolo_v3_aps_detections",
                       "frame_ev_pair_{}.txt".format(img_number))
dt_path = os.path.join(data_path, "..", "eccv_results",
                       "set_2", "conv_10_3_4",
                       "frame_ev_pair_{}.txt".format(img_number))

data = np.load(img_path)
img = data["img"]
ev_img = data["ev_img"]

# load gt
#  gt = np.loadtxt(gt_path, usecols=(1, 2, 3, 4))
gt = np.loadtxt(gt_path, usecols=(2, 3, 4, 5))
dt = np.loadtxt(dt_path, usecols=(2, 3, 4, 5))

if len(dt.shape) == 1:
    dt = [dt]
if len(gt.shape) == 1:
    gt = [gt]

plt.figure()
ax = plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.axis("off")
for box in gt:
    box = box.astype(int)
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax1 = plt.subplot(122)
ev_img = ev_img.mean(axis=0, keepdims=True)
ev_img = np.concatenate((ev_img, ev_img, ev_img), axis=0)
robust_max = ev_img.max()
robust_min = ev_img.min()
#  robust_max = np.percentile(ev_img.flatten(), 99)
#  robust_min = np.percentile(ev_img.flatten(), 1)
#  ev_img = np.clip(ev_img, robust_max, robust_min)
ev_img = (ev_img-robust_min)/(robust_max-robust_min)
plt.imshow(ev_img.transpose(1, 2, 0))
plt.axis("off")
for box in gt:
    box = box.astype(int)
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
for box in dt:
    box = box.astype(int)
    box[box < 0] = 0
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='blue', facecolor='none')
    ax1.add_patch(rect)

plt.show()
