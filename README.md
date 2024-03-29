# Network Grafting Algorithm

Code release for "Learning to Exploit Multiple Vision Modalities by Using Grafted Networks", ECCV 2020.

If you use this project, please cite:

- Y. Hu, T. Delbruck, S-C. Liu, "Learning to Exploit Multiple Vision Modalities by Using Grafted Networks" in The 16th European Conference on Computer Vision (ECCV), Online, 2020.

## Install `evtrans` locally

```
python setup.py develop
```

## Project structure

1. `evtrans` folder contains utility scripts and network definitions. Make sure you run `setup.py` to install this module locally.

2. `configs` folder contains pretrained network configurations and class definitions.

3. `scripts` folder contains:
    - `nmnist`: scripts that are related to N-MNIST classification.
    - `object_detection`: inference scripts related to thermal and event data experiments.
    - `prepare-data`: how to prepare data for N-MNIST, MVSEC and FLIR datasets.
    - `weights`: you should put the weights here. Small weight files are uploaded in this folder and a Google drive shared folder (see following).

## Run the script

1. Object Detection on Thermal Driving Dataset

    ```
    python export_thermal_yolo_results.py --val_data_dir /path/to/val/data --checkpoint /path/to/checkpoint.pt --detection_path /path/to/dump/prediction/result --conv_input_dim 1 --img_size 640 --cut_stage [1, 2, or 3] [--vis]
    ```

2. Car Detection on Event Camera Driving Dataset

    ```
    python export_ev_yolo_results.py --img_size 346 --val_data_dir /path/to/val/data --checkpoint /path/to/checkpoint.pt --detection_path /path/to/dump/prediction/result --conv_input_dim [3 or 10] --cut_stage [1, 2 or 3]
    ```

3. N-MNIST Classification

    ```
    python val_pt_nmnist.py --test_path /path/to/val/data --checkpoint /path/to/checkpoint.pt
    ```

## Trained weights

Some selected models are shared [here](https://drive.google.com/drive/folders/1ikGBtDfMlsu_BVDsyzyxSfVxOpnahz0L?usp=sharing).

This folder contains the original YOLOv3 pretrained weights and trained GN frontend for each task and configuration.

## Data

Data for evaluation is released [here](https://drive.google.com/drive/folders/1xVvveX9TF4Zoss-Crn4cUUK106UohfaY?usp=sharing).

## Alternative Downloads

If you cannot access above links, please go to Zenodo platform and download, use this [link](https://zenodo.org/record/4818421#.YK-8UHUzbmE)

## Contacts

Yuhuang Hu  
yuhuang.hu@ini.uzh.ch
