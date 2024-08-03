#!/bin/bash

conda activate banner-replacement
echo "Starting job at $(date)"
pwd

# Version of the NBJW model
NBJW_VERSION=3
WIDTH=1920
HEIGHT=1080
THRESHOLD=5
# Directory to save the inferences (ground truth and predictions)
SOURCE_DIR="inferences_${NBJW_VERSION}_${HEIGHT}_${WIDTH}/"
# Split to evaluate
SPLIT="valid"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the predictions zip file (camera parameters predicted by the NBJW model)
ZIP_NAME_IN="pred.zip"
# Name of the output predictions zip file (camera parameters after filtering)
ZIP_NAME_OUT="pred_wl.zip"
# Number of filtering layers to apply
N_LAYERS=3
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

# For window filter length in {3, 5, ..., 73, 75}
for OUTLIER_FILTER_WINDOW_LENGTH in {11..15..2}; do
    for SMOOTHING_FILTER_WINDOW_LENGTH in {19..27..2}; do
        echo "Camera parameters filtering"
        echo "Window filter length: $OUTLIER_FILTER_WINDOW_LENGTH"
        echo "Smoothing filter length: $SMOOTHING_FILTER_WINDOW_LENGTH"
        if [ $TEST = "True" ]; then
            python cam_params_filtering.py -t -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT --outlier_filter_window_length $OUTLIER_FILTER_WINDOW_LENGTH --smoothing_filter_window_length $SMOOTHING_FILTER_WINDOW_LENGTH
        else
            python cam_params_filtering.py --silent -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT --outlier_filter_window_length $OUTLIER_FILTER_WINDOW_LENGTH --smoothing_filter_window_length $SMOOTHING_FILTER_WINDOW_LENGTH
        fi

        echo "Evaluation"

        if [ $TEST = "True" ]; then
            python evalai_camera.py -t -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT
        else
            python evalai_camera.py --silent -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS --threshold $THRESHOLD --width $WIDTH --height $HEIGHT
        fi
    done
done

echo "Job finished at $(date)"