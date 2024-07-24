# Description:
#
#   -   Apply the given number of filtering layers to the camera parameters
#       for evaluation.

import argparse
import os
import sys
import json
import zipfile
import numpy as np
from tqdm import tqdm

sn_calibration_src_path = os.path.abspath("../camera_params_filtering/")
sys.path.append(sn_calibration_src_path)

from complete_camera_params_filtering import (  # type: ignore
    linear_interpolation,
    to_valid_cam_params,
    camParamsPerImage_to_camParamsPerType,
    camParamsPerType_to_camParamsPerImage,
)


def filter_zip_dir(zip_dir, zip_name_in, zip_name_out, length, n_layers):
    zipArchive = zipfile.ZipFile(os.path.join(zip_dir, zip_name_in), "r")
    zipJsons = zipArchive.namelist()
    zipJsons.sort()
    camParamsPerImage = []
    # Create a list of size "args.length" of camera parameters per image or None if absent
    zipJsonsImgNumbers = [int(j.split("_")[1].split(".")[0]) for j in zipJsons]
    for i in range(1, length + 1):
        if i in zipJsonsImgNumbers:
            camParamsPerImage.append(
                json.loads(
                    zipArchive.read(zipJsons[zipJsonsImgNumbers.index(i)]).decode(
                        "utf-8"
                    )
                )
            )
        else:
            camParamsPerImage.append(None)
    zipArchive.close()

    camParamsPerImage = np.array(camParamsPerImage)
    camParamsPerImage, isErroneousParams, ErroneousParamsPos = to_valid_cam_params(
        camParamsPerImage
    )
    camParamsPerType = camParamsPerImage_to_camParamsPerType(camParamsPerImage)

    if n_layers >= 1:
        camParamsPerType = linear_interpolation(
            camParamsPerType, isErroneousParams, ErroneousParamsPos
        )

    camParamsPerImage = camParamsPerType_to_camParamsPerImage(camParamsPerType)
    with zipfile.ZipFile(os.path.join(zip_dir, zip_name_out), "w") as zipFile:
        for i in range(length):
            zipFile.writestr(
                f"camera_{str(i+1).zfill(6)}.json", json.dumps(camParamsPerImage[i])
            )


def cpu_count():
    tmp = os.cpu_count()
    if tmp is None:
        return 1
    return tmp


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Path to the inference folder containing the groundtruth and the predictions per video/sequence",
        required=True,
    )
    parser.add_argument(
        "--zip_name_in",
        type=str,
        required=True,
        help="Input zip file name",
    )
    parser.add_argument(
        "--zip_name_out",
        type=str,
        required=True,
        help="Output zip file name",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        required=True,
        help="Number of frames in the video",
    )
    parser.add_argument(
        "-n",
        "--n_layers",
        type=int,
        required=True,
        help="Number of filtering layers to apply: 1 for linear interpolation,\
            2 for linear interpolation and outliers removal, 3 for linear\
            interpolation, outliers removal and smoothing",
    )
    default_workers = cpu_count() - 2 if cpu_count() > 2 else 1
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.length < 1:
        raise ValueError("Length must be greater than 0")
    if args.n_layers < 1 or args.n_layers > 3:
        raise ValueError("Number of layers must be 1, 2 or 3")

    directories = os.listdir(os.path.join(args.source, args.split))
    if args.test:
        directories.sort()
        directories = directories[:2]

    zip_dirs = [
        os.path.join(args.source, args.split, directory) for directory in directories
    ]

    for zip_dir in tqdm(zip_dirs, desc="Processing Dirs"):
        filter_zip_dir(
            zip_dir,
            args.zip_name_in,
            args.zip_name_out,
            args.length,
            args.n_layers,
        )
