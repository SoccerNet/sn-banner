# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from ast import arg
import os
import os.path as osp

from PIL import Image
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from tqdm import tqdm

classes = ["Outside billboards", "Billboard", "Goal net", "Occlusion"]
palette = [[0, 0, 0], [255, 255, 255], [0, 255, 0], [255, 0, 0]]

# Check if the dataset SoccerNet is already registered
if "SoccerNet" not in DATASETS:

    @DATASETS.register_module()
    class SoccerNet(BaseSegDataset):
        METAINFO = dict(classes=classes, palette=palette)

        def __init__(self, **kwargs):
            super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


def relativePathFromThisFile(path):
    return osp.join(osp.dirname(__file__), path)


def mask2former_inference(use_tta=False):
    # load config
    cfg = Config.fromfile(
        relativePathFromThisFile(
            "models/challenge_mask2former/challenge_mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py"
        )
    )
    cfg.launcher = "none"
    cfg.default_hooks.logger.interval = 15
    cfg.test_dataloader.batch_size = 1

    cfg.load_from = relativePathFromThisFile(
        "models/challenge_mask2former/best_mIoU_iter_10935.pth"
    )

    if use_tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    out = relativePathFromThisFile("../work_dir/masks")
    cfg.test_evaluator["output_dir"] = out

    cfg.data_root = relativePathFromThisFile("../work_dir")
    cfg.img_dir = "images"
    cfg.test_dataloader.dataset.data_prefix = dict(
        img_path=cfg.img_dir, seg_map_path=cfg.img_dir
    )
    cfg.test_dataloader.dataset.ann_file = relativePathFromThisFile(
        "../work_dir/ann_file.txt"
    )
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_evaluator.format_only = True
    cfg.work_dir = relativePathFromThisFile("../work_dir/logs/")

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()

    os.system(f"rm -r {cfg.work_dir}")


if __name__ == "__main__":
    mask2former_inference()
