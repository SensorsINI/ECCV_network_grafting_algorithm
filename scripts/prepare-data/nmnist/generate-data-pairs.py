"""Generate data pairs for NMNIST dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import os
import glob
import torch

import numpy as np

from evtrans.ncaltech_utils import load_events_bin
from evtrans import corrmaps

parser = argparse.ArgumentParser()

parser.add_argument("--mnist_path", type=str)
parser.add_argument("--nmnist_path", type=str)
parser.add_argument("--output_path", type=str)

args = parser.parse_args()

if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)

# load mnist
mnist_data, mnist_label = torch.load(args.mnist_path)
mnist_data = mnist_data.data.numpy()
mnist_label = mnist_label.data.numpy()

file_list = sorted(glob.glob("{}".format(args.nmnist_path)+"/*/*.*"))

for file_path in file_list:
    file_root = os.path.basename(file_path)[:-4]

    file_index = int(file_root)
    mnist_sample = np.zeros((32, 32), dtype=np.uint8)
    mnist_sample[2:30, 2:30] = mnist_data[file_index-1]

    # process image data
    all_y, all_x, all_p, all_t = load_events_bin(file_path)

    candidate_events = np.concatenate(
        (all_t[..., np.newaxis]/1e6,
         all_x[..., np.newaxis],
         all_y[..., np.newaxis],
         all_p[..., np.newaxis]), axis=1)

    # get voxel grid
    dvs_dens_map = corrmaps.events_to_voxel_grid(
        candidate_events.copy(), num_bins=3,
        width=34, height=34)

    # normalization
    nonzero_ev = (dvs_dens_map != 0)
    num_nonzeros = nonzero_ev.sum()

    # has events
    if num_nonzeros > 0:
        mean = dvs_dens_map.sum()/num_nonzeros
        stddev = np.sqrt((dvs_dens_map**2).sum()/num_nonzeros-mean**2)
        mask = nonzero_ev.astype("float32")
        dvs_dens_map = mask*(dvs_dens_map-mean)/stddev

    dvs_dens_map = dvs_dens_map[:, 1:33, 1:33]

    output_file_path = os.path.join(
        args.output_path, file_root+".npz")

    np.savez(output_file_path+".npz", ev_img=dvs_dens_map,
             img=mnist_sample[..., np.newaxis])

    print("Write to {}".format(output_file_path))
