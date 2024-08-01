# Credits:
#
#   -   https://github.com/SoccerNet/sn-calibration/blob/main/src/evalai_camera.py
#   -   https://github.com/mguti97/No-Bells-Just-Whistles/blob/main/sn_calibration/src/evalai_camera.py

import zipfile
import argparse
import numpy as np
import json
from tqdm import tqdm
import sys
import os
from multiprocessing import Pool

sn_calibration_src_path = os.path.abspath(
    "../No_Bells_Just_Whistles/sn_calibration/src/"
)
sys.path.append(sn_calibration_src_path)

from evaluate_camera import (  # type: ignore
    get_polylines,
    scale_points,
    evaluate_camera_prediction,
)
from evaluate_extremities import (  # type: ignore
    mirror_labels,
)

np.seterr(divide="ignore", invalid="ignore")


def evaluate_gt_pred_json(gt_pred_json_tuple):
    width = 1920
    height = 1080
    gt, prediction = gt_pred_json_tuple

    line_annotations = scale_points(gt, width, height)

    img_groundtruth = line_annotations

    img_prediction = get_polylines(prediction, width, height, sampling_factor=0.9)

    confusion1, _, _ = evaluate_camera_prediction(img_prediction, img_groundtruth, 5)

    confusion2, _, _ = evaluate_camera_prediction(
        img_prediction, mirror_labels(img_groundtruth), 5
    )

    accuracy1, accuracy2 = 0.0, 0.0
    if confusion1.sum() > 0:
        accuracy1 = confusion1[0, 0] / confusion1.sum()

    if confusion2.sum() > 0:
        accuracy2 = confusion2[0, 0] / confusion2.sum()

    if accuracy1 > accuracy2:
        accuracy = accuracy1
    else:
        accuracy = accuracy2

    return accuracy


def cpu_count():
    tmp = os.cpu_count()
    if tmp is None:
        return 1
    return tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation camera calibration task")

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Path to the inference folder containing the groundtruth and the predictions per video/sequence",
        required=True,
    )
    parser.add_argument(
        "--gt_zip_name",
        type=str,
        help="Groundtruth zip file name",
        required=True,
    )
    parser.add_argument(
        "--pred_zip_name",
        type=str,
        help="Prediction zip file name",
        required=True,
    )
    parser.add_argument("--split", type=str, required=True, help="Dataset split")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    default_workers = cpu_count() - 2 if cpu_count() > 2 else 1
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--silent", action="store_true")

    args = parser.parse_args()

    print("Number of workers:", args.workers)

    directories = os.listdir(os.path.join(args.source, args.split))
    if args.test:
        directories.sort()

    zip_dirs = [
        os.path.join(args.source, args.split, directory) for directory in directories
    ]

    total_frames = 0
    missed = 0
    gt_pred_json_tuples = []
    for zip_dir in zip_dirs:
        gt_zip = os.path.join(zip_dir, args.gt_zip_name)
        prediction_zip = os.path.join(zip_dir, args.pred_zip_name)
        gt_archive = zipfile.ZipFile(gt_zip, "r")
        prediction_archive = zipfile.ZipFile(prediction_zip, "r")
        gt_jsons = gt_archive.namelist()
        prediction_jsons = prediction_archive.namelist()
        for gt_json in gt_jsons:
            pred_name = f"camera_{gt_json}"

            total_frames += 1

            if pred_name not in prediction_jsons:
                missed += 1
                continue

            prediction = prediction_archive.read(pred_name)
            prediction = json.loads(prediction.decode("utf-8"))
            gt = gt_archive.read(gt_json)
            gt = json.loads(gt.decode("utf-8"))
            gt_pred_json_tuples.append((gt, prediction))

    if args.test:
        gt_pred_json_tuples = gt_pred_json_tuples[:24]

    accuracies = []
    print("total_frames", total_frames)
    with Pool(args.workers) as p:
        accuracies = list(
            tqdm(
                p.imap_unordered(evaluate_gt_pred_json, gt_pred_json_tuples),
                total=len(gt_pred_json_tuples),
                desc="Processing Frames",
                disable=args.silent,
            )
        )

    print("len(accuracies)", len(accuracies))
    completeness = (total_frames - missed) / total_frames
    mean_accuracies = np.mean(accuracies)
    finalScore = completeness * mean_accuracies
    print(f"Results for {args.split}")
    print(f"Completeness: {completeness}")
    print(f"Mean Accuracies: {mean_accuracies}")
    print(f"Final Score: {finalScore}")
