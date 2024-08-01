# Credits:
#
#   -   https://github.com/mguti97/No-Bells-Just-Whistles/blob/main/scripts/run_pipeline_sn22.sh
#
# Description:
#
#   -   This script is used to evaluate the first layer of filtering on the camera
#       parameters predicted by the NBJW model.
#       The first filtering layer linearly interpolates missing values in the camera parameters.

#!/bin/bash

# Set parameters

# Path to the inference directory
SOURCE_DIR="inferences2/"
# Split to evaluate
SPLIT="valid"
# Name of the ground truth zip file (pitch annotations from sn-gamestate dataset)
GT_ZIP_NAME="gt.zip"
# Name of the predictions zip file (camera parameters predicted by the NBJW model)
ZIP_NAME_IN="pred.zip"
# Name of the output predictions zip file (camera parameters after filtering)
ZIP_NAME_OUT="pred_final3.zip"
# Number of filtering layers to apply
N_LAYERS=3
# Length of the videos
VIDEO_LENGTH=750
# Number of concurrent processes to speed up the evaluation
N_WORKERS=14
# Set to True to inference and evaluate the 2 first videos only
TEST="True"

echo "Camera parameters filtering"

# Run inference script
if [ $TEST = "True" ]; then
    python cam_params_filtering.py -t -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT
else
    python cam_params_filtering.py -s $SOURCE_DIR --zip_name_in $ZIP_NAME_IN --zip_name_out $ZIP_NAME_OUT --workers $N_WORKERS -n $N_LAYERS -l $VIDEO_LENGTH --split $SPLIT
fi

echo "Evaluation"

# Run evaluation script
if [ $TEST = "True" ]; then
    python evalai_camera.py -t -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS
else
    python evalai_camera.py -s $SOURCE_DIR --split $SPLIT --gt_zip_name $GT_ZIP_NAME --pred_zip_name $ZIP_NAME_OUT --workers $N_WORKERS
fi
