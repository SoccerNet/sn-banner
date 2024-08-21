#!/bin/bash

echo "Starting job at $(date)"
pwd

# Version of the NBJW model
NBJW_VERSION=3
WIDTH=1920
HEIGHT=1080
THRESHOLD=5
# Number of filtering layers to apply
N_LAYERS=3 # 1, 2 or 3
# Number of concurrent processes to speed up the evaluation
N_WORKERS=16
# Set to True to inference and evaluate the 2 first videos only
TEST="True"
# Split to evaluate
SPLIT="valid"
# Directory to save the inferences (ground truth and predictions)
SOURCE_DIR="inferences_${NBJW_VERSION}_${HEIGHT}_${WIDTH}/"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the output predictions zip file (camera parameters after filtering)
PRED_ZIP_NAME="pred.zip"
FILTERED_PRED_ZIP_NAME="pred_filtered_${N_LAYERS}_layers.zip"

# print all parameters
echo "NBJW_VERSION: $NBJW_VERSION"
echo "WIDTH: $WIDTH"
echo "HEIGHT: $HEIGHT"
echo "THRESHOLD: $THRESHOLD"
echo "SOURCE_DIR: $SOURCE_DIR"
echo "SPLIT: $SPLIT"
echo "TEST: $TEST"

echo "Evaluation"

if [ $TEST = "True" ]; then
    python evalai_camera_per_video.py -t -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT --pred_filtered_zip_name $FILTERED_PRED_ZIP_NAME
else
    python evalai_camera_per_video.py -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $PRED_ZIP_NAME --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT --pred_filtered_zip_name $FILTERED_PRED_ZIP_NAME
fi

echo "Job finished at $(date)"
