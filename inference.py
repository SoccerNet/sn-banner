import os, json
import numpy as np
from camera_calibration.filters import (
    linear_interpolation,
    to_valid_cam_params,
    camParamsPerImage_to_camParamsPerType,
    camParamsPerType_to_camParamsPerImage,
    outliers_remover,
    camParamsSmoothing,
)
from semantic_segmentation.filter import keep_biggest_blob
from tqdm import tqdm
import cv2
from compositing.utils import compute_banner_model_params, composite_logo_into_video
from camera_calibration.No_Bells_Just_Whistles.inference import process_image_sequence
from copy import deepcopy
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str,
        help="Path to the video file",
    )
    parser.add_argument(
        "logo",
        type=str,
        help="Path to the logo image",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to speed up the inference. Can consumme a lot of memory",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.75,
        help="Speed of banner movement",
    )
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Use this flag if the video is a sequence of images",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logoPath = args.logo
    videoPath = args.video
    nWorkers = args.n_workers
    speed = args.speed
    isSequence = args.sequence

    if isSequence:
        imgFileNames = os.listdir(videoPath)
        imgFileNames.sort()
        nFrames = len(imgFileNames)
        # All files are names "str(i+1).zfill(6).jpg" for i in range(nFrames), convert and rename them to "str(i).zfill(6).png"
        for i, imgFileName in enumerate(tqdm(imgFileNames, desc="Renaming images")):
            img = cv2.imread(os.path.join(videoPath, imgFileName))
            cv2.imwrite("work_dir/images/" + str(i).zfill(6) + ".png", img)
        frameSample = cv2.imread("work_dir/images/000000.png")
        imgWidth = frameSample.shape[1]
        imgHeight = frameSample.shape[0]
        fps = 25
    else:
        # Check that the video file exists and it is a video file
        if not os.path.isfile(videoPath):
            raise FileNotFoundError(f"File not found: {videoPath}")
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            raise ValueError(f"File is not a video file: {videoPath}")
        imgWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imgHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        for i in tqdm(range(nFrames), desc="Extracting frames from video"):
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Error reading video file")
            cv2.imwrite("work_dir/images/" + str(i).zfill(6) + ".png", frame)

    with open("work_dir/ann_file.txt", "w") as f:
        for i in range(nFrames):
            f.write(f"{str(i).zfill(6)}\n")
        # remove the last newline character
        f.seek(f.tell() - 1)

    # Run the semantic segmentation model
    if args.tta:
        os.system("conda run -n mmseg python semantic_segmentation/inference.py --tta")
    else:
        os.system("conda run -n mmseg python semantic_segmentation/inference.py")

    imgPath = "work_dir/images/"
    maskPath = "work_dir/masks/"

    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
    if not os.path.exists(maskPath):
        os.mkdir(maskPath)
    if not os.path.exists("work_dir/output"):
        os.mkdir("work_dir/output")

    # Camera calibration
    process_image_sequence(
        imgPath,
        "work_dir/",
        nFrames,
        weights_kp="./camera_calibration/No_Bells_Just_Whistles/SV_kp",
        weights_line="./camera_calibration/No_Bells_Just_Whistles/SV_lines",
    )

    with open("work_dir/cam_params.json", "r") as f:
        camParamsPerImage = json.load(f)

    camParamsPerImage = np.array(camParamsPerImage)
    camParamsPerImage, isErroneousParams, ErroneousParamsPos = to_valid_cam_params(
        camParamsPerImage
    )
    camParamsPerType = camParamsPerImage_to_camParamsPerType(camParamsPerImage)
    camParamsPerType = linear_interpolation(
        camParamsPerType, isErroneousParams, ErroneousParamsPos
    )
    camParamsPerType = outliers_remover(
        camParamsPerType,
        isErroneousParams,
        ErroneousParamsPos,
    )
    basicCamParamsPerType = deepcopy(camParamsPerType)
    camParamsPerType = camParamsSmoothing(camParamsPerType)
    camParamsPerImage = camParamsPerType_to_camParamsPerImage(camParamsPerType)

    masksFileNames = os.listdir(maskPath)
    for maskFileName in tqdm(
        masksFileNames, desc="Filtering semantic segmentation masks"
    ):
        filteredMask = keep_biggest_blob(os.path.join(maskPath, maskFileName))
        cv2.imwrite(os.path.join(maskPath, maskFileName), filteredMask)

    bannersObjPts, bannerHeight = compute_banner_model_params(
        camParamsPerImage, maskPath, imgWidth, imgHeight, nWorkers, nFrames
    )

    # get video name from videoPath
    if isSequence:
        videoName = videoPath.split("/")[-3]
    else:
        videoName = videoPath.split("/")[-1].split(".")[0]
    outputVideoName = f"output_{videoName}.mp4"
    print("outputVideoName: ", outputVideoName)

    composite_logo_into_video(
        logoPath,
        imgPath,
        maskPath,
        camParamsPerImage,
        imgWidth,
        imgHeight,
        nWorkers,
        nFrames,
        fps,
        speed,
        bannersObjPts,
        bannerHeight,
        outputVideoName,
    )

    # print that the ouput video is ready and in the directory work_dir
    print(f"Output video is ready: {outputVideoName}")
    print("You can find it in the directory work_dir")
