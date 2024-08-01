# Credits:
#
#   -   https://github.com/mguti97/No-Bells-Just-Whistles/blob/main/scripts/run_pipeline_sn22.sh
#
# Description:
#
#   -   This script is used to evaluate a second modification of the camera calibration
#       model No Bells Just Whistles (NBJW) on the SoccerNet Game State (sn-gamestate)
#       dataset.

#!/bin/bash

# Set parameters

# Directory to the sn-gamestate dataset
ROOT_DIR="../../camera-calibration/sn-gamestate/data/SoccerNetGS"
# Split to evaluate
SPLIT="valid"
# Path to the configuration file for the keypoint detection model
CFG="../No_Bells_Just_Whistles/config/hrnetv2_w48.yaml"
# Path to the configuration file for the line detection model
CFG_L="../No_Bells_Just_Whistles/config/hrnetv2_w48_l.yaml"
# Path to the weights of the keypoint detection model
WEIGHTS_KP="../No_Bells_Just_Whistles/SV_kp"
# Path to the weights of the line detection model
WEIGHTS_L="../No_Bells_Just_Whistles/SV_lines"
# Directory to save the inferences (ground truth and predictions)
SAVE_DIR="inferences3/"
# Device to use for inference
DEVICE="cuda:0"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the predictions zip file (camera parameters predicted by the NBJW model)
PRED_ZIP_NAME="pred.zip"
# Version of the NBJW model
NBJW_VERSION=3
# Number of concurrent processes to speed up the evaluation
N_WORKERS=14
# Set to True to inference and evaluate the 2 first videos only
TEST="False"

mkdir -p $SAVE_DIR/$SPLIT

# Run inference script
if [ $TEST = "True" ]; then
    python inference_NBJW.py -t --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE -v $NBJW_VERSION --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME
else
    python inference_NBJW.py --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE -v $NBJW_VERSION --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME
fi

# Run evaluation script
if [ $TEST = "True" ]; then
    python evalai_camera.py -t -s $SAVE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --workers $N_WORKERS
else
    python evalai_camera.py -s $SAVE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --workers $N_WORKERS
fi
