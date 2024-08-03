#!/bin/bash

# Activate your Anaconda environment
conda activate banner-replacement

echo "Starting job at $(date)"
pwd

# Version of the NBJW model
NBJW_VERSION=3
WIDTH=1920
HEIGHT=1080
THRESHOLD=20 #5, 10 or 20 pixels are the most common thresholds
# Number of filtering layers to apply
N_LAYERS=3  # 1, 2 or 3
# Split to evaluate
SPLIT="test"
# Directory to save the inferences (ground truth and predictions)
SOURCE_DIR="inferences_${NBJW_VERSION}_${HEIGHT}_${WIDTH}/"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the predictions zip file (camera parameters predicted by the NBJW model)
ZIP_NAME_IN="pred.zip"
# Name of the output predictions zip file (camera parameters after filtering)
ZIP_NAME_OUT="pred_filtered_${N_LAYERS}_layers.zip"
# Length of the videos
VIDEO_LENGTH=750
# Number of concurrent processes to speed up the evaluation
N_WORKERS=220
# Set to True to inference and evaluate the 2 first videos only
TEST="False"

# print all parameters
echo "NBJW_VERSION: $NBJW_VERSION"
echo "WIDTH: $WIDTH"
echo "HEIGHT: $HEIGHT"
echo "THRESHOLD: $THRESHOLD"
echo "SOURCE_DIR: $SOURCE_DIR"
echo "SPLIT: $SPLIT"
echo "TEST: $TEST"

echo "Camera parameters filtering"

# Run inference script
if [ $TEST = "True" ]; then
    python cam_params_filtering.py -t -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT
else
    python cam_params_filtering.py -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT
fi

echo "Evaluation"

if [ $TEST = "True" ]; then
    python evalai_camera.py -t -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT
else
    python evalai_camera.py -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT
fi

echo "Job finished at $(date)"