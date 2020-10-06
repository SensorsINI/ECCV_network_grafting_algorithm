"""Decode ground truth data for evaluation.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--gt_file", type=str)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()

with open(args.gt_file) as f:
    gt_data = json.load(f)

gt_class = gt_data["categories"]
gt_ann = gt_data["annotations"]
gt_img = gt_data["images"]

if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

id2name_map = {}
id2class_map = {}

for class_name in gt_class:
    id2class_map[class_name["id"]] = class_name["name"]

for img_desc in gt_img:
    img_id = img_desc["id"]
    out_file_name = os.path.basename(img_desc["file_name"])[:-5]+".txt"
    out_file_name = os.path.join(args.save_path, out_file_name)

    id2name_map[img_id] = out_file_name

# write down the detections
for ann in gt_ann:
    img_id = ann["image_id"]
    out_file = id2name_map[img_id]

    bbox = ann["bbox"]
    class_name = id2class_map[ann["category_id"]]

    with open(out_file, "a+") as f:
        out_line = "{} {} {} {} {}\n".format(
            class_name, bbox[0], bbox[1], bbox[0]+bbox[2],
            bbox[1]+bbox[3])

        f.write(out_line)
