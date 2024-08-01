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


def computeBConfMat(pred, gt, nClasses, distances):
    confMatPerDist = np.zeros(
        (len(distances), nClasses + 1, nClasses + 1), dtype=np.int64
    )
    if np.any(np.isinf(distances)):
        confMatPerDist[-1] += confusion_matrix(
            gt.flatten(), pred.flatten(), labels=np.arange(nClasses + 1)
        )

    distances = [int(d) for d in distances if not np.isinf(d)]
    if len(distances) == 0:
        return confMatPerDist

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

    for i, d in enumerate(distances):
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

        confMatPerDist[i] += confusion_matrix(
            boundaryGt.flatten(), boundaryPred.flatten(), labels=np.arange(nClasses + 1)
        )

    return confMatPerDist


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
        "--distances",
        type=str,
        default="1,3,5,10,inf",
        help="Distances to compute the boundary confusion matrix. Inf is equivalent to a regular confusion matrix computation.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers to speed up the computation. Should be equal to the number of available cores since each process will use two threads.",
    )

    args = parser.parse_args()
    return args


def computeBConfMatWrapper(imgFileName):
    pred = torch.from_numpy(np.array(Image.open(predPath + imgFileName))).long()
    gt = torch.from_numpy(np.array(Image.open(gtPath + imgFileName))).long()
    return computeBConfMat(pred, gt, nClasses, distances)


if __name__ == "__main__":
    args = parse_args()

    predPath = args.pred_path
    gtPath = args.gt_path
    distances = [float(d) for d in args.distances.split(",")]
    nWorkers = args.n_workers

    nClasses = 4
    imgFileNames = os.listdir(predPath)

    with Pool(nWorkers) as p:
        confMatPerDist = list(
            tqdm(
                p.imap_unordered(computeBConfMatWrapper, imgFileNames),
                total=len(imgFileNames),
            )
        )

    confMatPerDist = np.sum(confMatPerDist, axis=0)
    np.save(args.output_name, confMatPerDist)
