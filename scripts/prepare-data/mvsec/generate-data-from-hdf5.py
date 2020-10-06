
"""Generate parallel data pairs for DDD19 dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import argparse

import numpy as np

import h5py
from evtrans import corrmaps

parser = argparse.ArgumentParser()
# number of events in event volume
parser.add_argument("--num-events", "-n", type=int,
                    default=10000)
# number of slices in the event volume
parser.add_argument("--num-slices", "-s", type=int,
                    default=3)
# maximum number of frames per dataset
parser.add_argument("--max-frames", "-m", type=int,
                    default=-1)

# single file
parser.add_argument("--data-file", "-d", type=str,
                    default="")
parser.add_argument("--data-output-dir", "-o", type=str,
                    default="")
parser.add_argument("--save-events", type=bool, default=False)
args = parser.parse_args()

data_file_paths = [args.data_file]
data_dir_path = args.data_output_dir

if not os.path.isdir(data_dir_path):
    os.makedirs(data_dir_path)

num_frame_generated = 0
for data_path in data_file_paths:
    dataset = h5py.File(data_path, "r")
    num_frames = dataset["davis/left"]["image_raw"].shape[0]
    num_events = dataset["davis/left"]["events"].shape[0]
    num_export_frames = num_frames \
        if args.max_frames == -1 else args.max_frames // len(data_file_paths)

    print("Number of frames {}".format(num_frames))
    print("Number of targeted exported frames {}".format(num_export_frames))
    print("Number of events {}".format(num_events))

    for frame_idx in range(1, num_frames-1,
                           max(num_frames//num_export_frames//2, 1)):
        print("Generating Frame ID: {}".format(frame_idx))

        # the closest event idx
        event_id = dataset["davis/left"][
            "image_raw_event_inds"][frame_idx]

        # select events
        ev_vol_end_idx = min(event_id+args.num_events//2, num_events)
        ev_vol_start_idx = max(event_id-args.num_events//2, 0)

        # condition on too few events
        if (ev_vol_end_idx - ev_vol_start_idx) < args.num_events:
            print("Bad frame {}, not enough events".format(
                frame_idx))
            continue

        # if the start and the end reached another frame, then we dump this
        # means too few events here
        ev_start_ts = dataset["davis/left"]["events"][ev_vol_start_idx, 2]
        ev_end_ts = dataset["davis/left"]["events"][ev_vol_end_idx, 2]

        curr_frame_ts = dataset["davis/left"]["events"][event_id, 2]
        next_frame_ts = dataset["davis/left"]["events"][
            dataset["davis/left"]["image_raw_event_inds"][frame_idx+1], 2]
        pre_frame_ts = dataset["davis/left"]["events"][
            dataset["davis/left"]["image_raw_event_inds"][frame_idx-1], 2]

        if ev_start_ts < pre_frame_ts or ev_end_ts > next_frame_ts:
            print("Bad frame {}, too long".format(
                frame_idx))
            continue

        # select events
        candidate_events = dataset["davis/left"]["events"][
            ev_vol_start_idx:ev_vol_end_idx][()]
        # swap time
        candidate_events = np.append(
            candidate_events[:, 2][..., np.newaxis],
            candidate_events[:, [0, 1, 3]],
            axis=1)

        # get voxel grid
        dvs_dens_map = corrmaps.events_to_voxel_grid(
            candidate_events.copy(), num_bins=args.num_slices,
            width=346, height=260)

        # normalization
        nonzero_ev = (dvs_dens_map != 0)
        num_nonzeros = nonzero_ev.sum()

        # has events
        if num_nonzeros > 0:
            mean = dvs_dens_map.sum()/num_nonzeros
            stddev = np.sqrt((dvs_dens_map**2).sum()/num_nonzeros-mean**2)
            mask = nonzero_ev.astype("float32")
            dvs_dens_map = mask*(dvs_dens_map-mean)/stddev

        # save the output file
        output_path = os.path.join(
            data_dir_path, "frame_ev_pair_{:05d}".format(
                num_frame_generated+1))

        # prepare data
        frame = dataset["davis/left"]["image_raw"][
            frame_idx][()]

        if args.save_events is False:
            np.savez(output_path+".npz", ev_img=dvs_dens_map,
                     img=frame)
        else:
            np.savez(output_path+".npz", ev_img=dvs_dens_map,
                     img=frame, events=candidate_events)

        num_frame_generated += 1
        print("Generated data for frame {} at {}".format(
            frame_idx, num_frame_generated))

        if num_frame_generated == num_export_frames*len(data_file_paths):
            print("Data generation for this dataset is done")
            quit()
