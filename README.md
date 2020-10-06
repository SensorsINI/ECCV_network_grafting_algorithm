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
    - `weights`: you should put the weights here. Small weight files are uploaded in this folder, larger weight files (>100MB) will be provided through a downloadable link.

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


## Trained weights (to update)


## Contacts

Yuhuang Hu  
yuhuang.hu@ini.uzh.ch
