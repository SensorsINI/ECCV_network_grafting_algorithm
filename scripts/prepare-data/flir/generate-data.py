"""Generate training and val data for FLIR ADAS dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import glob

import os
import argparse

import numpy as np

from skimage.io import imread
from skimage.transform import rescale

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# maximum number of frames per dataset
parser.add_argument("--max-frames", "-m", type=int,
                    default=-1)

# single file
parser.add_argument("--data-file", "-d", type=str)
parser.add_argument("--data-output-dir", "-o", type=str)

parser.add_argument("--show", action="store_true")

args = parser.parse_args()

data_file_paths = args.data_file
data_dir_path = args.data_output_dir

if not os.path.isdir(data_dir_path):
    os.makedirs(data_dir_path)

rgb_path = os.path.join(args.data_file, "RGB")
thermal_path = os.path.join(args.data_file, "thermal_8_bit")

# get the list of the file
file_list = sorted(glob.glob("{}".format(rgb_path)+"/*.jpg"))

img_pair_counter = 0

for file_path in file_list:
    base_name = os.path.basename(file_path)[:-4]
    thermal_file_path = os.path.join(thermal_path, base_name+".jpeg")
    if not os.path.isfile(thermal_file_path):
        print("No such pair {}".format(base_name))
        continue

    # load image
    rgb_img = imread(file_path)
    if rgb_img.shape[0] != 1600:
        # skip image that does not fit
        print("Skip smaller image {}.".format(file_path))
        continue

    if rgb_img.ndim != 3:
        # skip image that does not fit
        print("Bad image {}.".format(file_path))
        continue

    # load thermal image
    thermal_img = imread(thermal_file_path)

    # align them together
    rgb_img = rescale(rgb_img, 0.4, multichannel=True)
    rgb_img = rgb_img[
        58:58+thermal_img.shape[0], 65:65+thermal_img.shape[1], :]

    rgb_img = (rgb_img*255).astype(np.uint8)

    # show image
    if args.show is True:
        plt.figure()
        plt.subplot(121)
        plt.imshow(rgb_img)

        plt.subplot(122)
        plt.imshow(thermal_img)

        plt.show()

    # write image pair to new directory
    output_path = os.path.join(
        data_dir_path, "frame_thermal_pair_{:05d}".format(
            img_pair_counter+1))
    np.savez(output_path+".npz", ev_img=thermal_img,
             img=rgb_img)

    print("Saved frame thermal pair at {}".format(output_path))

    img_pair_counter += 1
    if img_pair_counter == args.max_frames:
        break
