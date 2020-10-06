"""Show prediction.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import
import os

import numpy as np

from skimage.io import imread
from skimage.transform import rescale

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# image path
img_number = "09245"
data_path = os.path.join(os.environ["HOME"], "data", "FLIR_ADAS_1_3")
file_path = os.path.join(
    data_path, "val", "RGB", "FLIR_{}.jpg".format(img_number))
thermal_file_path = os.path.join(
    data_path, "val", "thermal_8_bit", "FLIR_{}.jpeg".format(img_number))

gt_path = os.path.join(data_path, "frame_pretrained",
                       "FLIR_{}.txt".format(img_number))
dt_path = os.path.join(data_path, "old_results", "thermal_3_4",
                       "FLIR_{}.txt".format(img_number))


# load image
rgb_img = imread(file_path)
if rgb_img.shape[0] != 1600:
    # skip image that does not fit
    print("Skip smaller image {}.".format(file_path))

if rgb_img.ndim != 3:
    # skip image that does not fit
    print("Bad image {}.".format(file_path))

# load thermal image
thermal_img = imread(thermal_file_path)[..., np.newaxis]
thermal_img = np.concatenate((thermal_img, thermal_img, thermal_img), axis=2)

# align them together
rgb_img = rescale(rgb_img, 0.4, multichannel=True)
rgb_img = rgb_img[
    58:58+thermal_img.shape[0], 65:65+thermal_img.shape[1], :]

rgb_img = (rgb_img*255).astype(np.uint8)

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
plt.imshow(rgb_img)
plt.axis("off")
for box in gt:
    box = box.astype(int)
    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax1 = plt.subplot(122)
plt.imshow(thermal_img)
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
