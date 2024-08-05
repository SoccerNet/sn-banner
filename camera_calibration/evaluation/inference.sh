#!/bin/bash

echo "Starting job at $(date)"
pwd

# Set parameters

# Directory to the sn-gamestate dataset
ROOT_DIR="data/SoccerNetGS/"
# Split to evaluate
SPLIT="test"
# Version of the NBJW model
NBJW_VERSION=3
WIDTH=1920
HEIGHT=1080
# Set to True to inference and evaluate the 2 first videos only
TEST="True"
# Path to the configuration file for the keypoint detection model
CFG="../No_Bells_Just_Whistles/config/hrnetv2_w48.yaml"
# Path to the configuration file for the line detection model
CFG_L="../No_Bells_Just_Whistles/config/hrnetv2_w48_l.yaml"
# Path to the weights of the keypoint detection model
WEIGHTS_KP="../No_Bells_Just_Whistles/SV_kp"
# Path to the weights of the line detection model
WEIGHTS_L="../No_Bells_Just_Whistles/SV_lines"
# Device to use for inference
DEVICE="cuda:0"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the predictions zip file (camera parameters predicted by the NBJW model)
PRED_ZIP_NAME="pred.zip"
# Directory to save the inferences (ground truth and predictions)
SAVE_DIR="inferences_${NBJW_VERSION}_${HEIGHT}_${WIDTH}/"

# print parameters
echo "SPLIT: $SPLIT"
echo "NBJW_VERSION: $NBJW_VERSION"
echo "WIDTH: $WIDTH"
echo "HEIGHT: $HEIGHT"
echo "SAVE_DIR: $SAVE_DIR"
echo "TEST: $TEST"

mkdir -p $SAVE_DIR/$SPLIT

# Run inference script
if [ $TEST = "True" ]; then
    # python inference_NBJW.py -t --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE -v $NBJW_VERSION --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --width $WIDTH --height $HEIGHT
    python inference_NBJW.py -t --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE -v $NBJW_VERSION --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --width $WIDTH --height $HEIGHT
else
    python inference_NBJW.py --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE -v $NBJW_VERSION --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --width $WIDTH --height $HEIGHT
fi

echo "Job finished"
