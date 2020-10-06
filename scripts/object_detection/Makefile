# Makefile for experiments
# Author: Yuhuang Hu
# Email : yuhuang.hu@ini.uzh.ch

validate-ev-yolo-script-test:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/ilsvrc/selected_data/data/val_final --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_script_test/checkpoints/checkpoint_45.pt --cut_stage 2 --detection_path ${HOME}/data/ilsvrc/selected_data/ann/ev_yolo_v3_dvs_3_detections --r1

validate-ev-yolo-combine-event-frame-test:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/DDD19/val_data_final --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_combine_event_frame_test/checkpoints/checkpoint_50.pt --cut_stage 2 --detection_path ${HOME}/data/DDD19/val_data_final/ev_yolo_v3_aps_dvs_detections

validate-ev-yolo-combine-event-frame-test-2:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/DDD19/val_data_final --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_combine_event_frame_test_2/checkpoints/checkpoint_100.pt --cut_stage 2 --detection_path ${HOME}/data/DDD19/val_data_final/ev_yolo_v3_aps_dvs_detections_2

validate-ev-yolo-mvsec-test:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/mvsec/val_data_final --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_mvsec_test/checkpoints/checkpoint_94.pt --cut_stage 2 --detection_path ${HOME}/data/mvsec/val_data_final/ev_yolo_v3_dvs_10_detections --r1

validate-ev-yolo-mvsec-objects-test:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/mvsec/clock --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_mvsec_test/checkpoints/checkpoint_94.pt --cut_stage 2 --detection_path ${HOME}/data/mvsec/clock/ev_yolo_v3_dvs_10_detections --r1
# --vis --output_save_path ${HOME}/data/mvsec/spoon_output
#
validate-ev-yolo-mvsec-objects-difficult-test:
	python ./export_ev_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/mvsec/laptop_blur --checkpoint ${HOME}/data/event_transfer/train_ev_yolo_mvsec_test/checkpoints/checkpoint_94.pt --cut_stage 2 --conv_input_dim 3 --detection_path ${HOME}/data/mvsec/laptop_blur/ev_yolo_v3_dvs_10_detections --r1
# --vis --output_save_path ${HOME}/data/mvsec/spoon_output

validate-thermal-yolo-test:
	python ./export_thermal_yolo_results.py --img_size 416 --val_data_dir ${HOME}/data/FLIR_ADAS_1_3/video_pair --checkpoint ${HOME}/data/event_transfer/train_thermal_yolo_test/checkpoints/checkpoint_50.pt --cut_stage 2 --detection_path ${HOME}/data/FLIR_ADAS_1_3/val_pair/thermal_yolo_detections --vis --output_save_path ${HOME}/data/FLIR_ADAS_1_3/thermal_val_output
