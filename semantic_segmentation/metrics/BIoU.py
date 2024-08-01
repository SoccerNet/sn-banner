import os
import pandas as pd
import torch
import numpy as np
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE
from tqdm import tqdm
from PIL import Image
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
import argparse

np.seterr(divide="ignore", invalid="ignore")


def computeBIoU(pred, gt):
    BIoUs = np.zeros((maxDist, nClasses), dtype=np.float32)

    predOneHot = (
        torch.nn.functional.one_hot(pred, 4).permute(2, 0, 1).numpy().astype(np.uint8)
    )
    gtOneHot = (
        torch.nn.functional.one_hot(gt, 4).permute(2, 0, 1).numpy().astype(np.uint8)
    )
    distPredOneHot = np.array(
        [distanceTransform(binImg, DIST_L2, DIST_MASK_PRECISE) for binImg in predOneHot]
    )
    distGtOneHot = np.array(
        [distanceTransform(binImg, DIST_L2, DIST_MASK_PRECISE) for binImg in gtOneHot]
    )

    for d in range(1, maxDist + 1):
        boundaryPredOneHot = np.bitwise_and(distPredOneHot <= d, predOneHot)
        boundaryGtOneHot = np.bitwise_and(distGtOneHot <= d, gtOneHot)
        # concatenate np.ones_like(boundaryPredOneHot[0]) at the end of the array to account for the background class
        boundaryPredOneHot = np.concatenate(
            (boundaryPredOneHot, np.ones_like(boundaryPredOneHot[0])[np.newaxis]),
            axis=0,
        )
        boundaryGtOneHot = np.concatenate(
            (boundaryGtOneHot, np.ones_like(boundaryGtOneHot[0])[np.newaxis]), axis=0
        )
        boundaryPred = np.argmax(boundaryPredOneHot, axis=0)
        boundaryGt = np.argmax(boundaryGtOneHot, axis=0)

        cm = confusion_matrix(
            boundaryGt.flatten(), boundaryPred.flatten(), labels=np.arange(nClasses + 1)
        )
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        BIoUs[d - 1] = (tp / (tp + fp + fn))[:nClasses]

    return BIoUs


def computeBIoUWrapper(imgFileName):
    pred = torch.from_numpy(np.array(Image.open(predPath + imgFileName))).long()
    gt = torch.from_numpy(np.array(Image.open(gtPath + imgFileName))).long()
    return computeBIoU(pred, gt)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "pred_path",
        type=str,
        help="Path to the predictions",
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="Path to the ground truth",
    )
    # output_name
    parser.add_argument(
        "output_name",
        type=str,
        help="Name of the output file",
    )
    # distances to compute the boundary confusion matrix
    parser.add_argument(
        "--max_distance",
        type=int,
        default=50,
        help="Maximum distance to compute the BIoU",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    predPath = args.pred_path
    gtPath = args.gt_path
    maxDist = args.max_distance
    nWorkers = args.n_workers
    nClasses = 4

    imgFileNames = os.listdir(predPath)

    with Pool(nWorkers / 2) as p:
        BIoUs = list(
            tqdm(
                p.imap_unordered(computeBIoUWrapper, imgFileNames),
                total=len(imgFileNames),
            )
        )
    BIoUs = np.mean(BIoUs, axis=0)
    np.save(args.output_name, BIoUs)
