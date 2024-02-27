# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.registry import TRANSFORMS
from mmseg.registry import DATASETS
from mmcv.transforms.base import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmcv.transforms.utils import cache_randomness
from typing import Tuple, Union
import numpy as np

old_classes = [
    "Outside billboards",
    "Billboard",
    "Field player",
    "Goalkeeper",
    "Referee",
    "Assistant referee",
    "Other human",
    "Ball",
    "Goal post",
    "Goal net",
    "Net post",
    "Cross-bar",
    "Corner flag",
    "Assistant flag",
    "Microphone",
    "Camera",
    "Other object",
    "Don't care",
]
old_palette = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [64, 64, 64],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [255, 128, 0],
    [192, 192, 192],
    [128, 128, 128],
]

new_classes = ["Outside billboards", "Billboard", "Goal net"]
new_palette = [old_palette[old_classes.index(c)] for c in new_classes]

# Check if the dataset SoccerNet is already registered
if "SoccerNet" not in DATASETS:

    @DATASETS.register_module()
    class SoccerNet(BaseSegDataset):
        METAINFO = dict(classes=new_classes, palette=new_palette)

        def __init__(self, **kwargs):
            super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


new_classes = ["Outside billboards", "Goal net"]
new_palette = [old_palette[old_classes.index(c)] for c in new_classes]

# Check if the dataset SoccerNetv2Gl is already registered
if "SoccerNetv2Gl" not in DATASETS:

    @DATASETS.register_module()
    class SoccerNetv2Gl(BaseSegDataset):
        METAINFO = dict(classes=new_classes, palette=new_palette)

        def __init__(self, **kwargs):
            super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


new_classes = ["Outside billboards", "Billboard"]
new_palette = [old_palette[old_classes.index(c)] for c in new_classes]

# Check if the dataset SoccerNetv2Bb is already registered
if "SoccerNetv2Bb" not in DATASETS:

    @DATASETS.register_module()
    class SoccerNetv2Bb(BaseSegDataset):
        METAINFO = dict(classes=new_classes, palette=new_palette)

        def __init__(self, **kwargs):
            super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


@TRANSFORMS.register_module()
class CustomTransform(BaseTransform):
    """Random crop the image & seg.

    Required Keys:

    - img
    - gt_seg_map

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (h, w). If set to an integer, then cropping
            width and height are equal to this integer.
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored. Default: 255
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        cat_max_ratio: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), "The expected crop_size is an integer, or a tuple containing two "
        "intergers"

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results["img"]
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for i in range(100):
                seg_temp = self.crop(results["gt_seg_map"], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    print_log(
                        "RandomCrop: cat_max_ratio is satisfied after {} trials.".format(
                            i + 1
                        ),
                    )
                    break
                crop_bbox = generate_crop_bbox(img)
                # If this is the last trial, print a warning
                if i == 99:
                    print_log(
                        "RandomCrop: cat_max_ratio is too small to be satisfied.",
                    )

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get("seg_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    # Add an option '--log' to specify after how many iterations the log
    # should be printed
    parser.add_argument(
        "--log",
        type=int,
        default=0,
        help="the interval of iterations to log the training status",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.log > 0:
        cfg.default_hooks.logger.interval = args.log
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume training
    cfg.resume = args.resume

    if args.resume:
        print("Resume training")
    else:
        # Check if the work_dir exists
        if osp.exists(cfg.work_dir):
            # Print directories in alphebetical descending order
            dirEntries = os.listdir(cfg.work_dir)
            files, dirs = [], []
            for entry in dirEntries:
                if osp.isfile(osp.join(cfg.work_dir, entry)):
                    files.append(entry)
                else:
                    dirs.append(entry)
            dirs.sort(reverse=True)
            lastDir = dirs[0] if len(dirs) > 0 else None
            if lastDir is not None:
                # Move all the files in the last directory to the work_dir
                # Except for the 'train.log' file
                for f in files:
                    if f == "train.log":
                        continue
                    if f == "trainn.log":
                        os.rename(
                            osp.join(cfg.work_dir, f),
                            osp.join(cfg.work_dir, lastDir, "train.log"),
                        )
                    else:
                        os.rename(
                            osp.join(cfg.work_dir, f),
                            osp.join(cfg.work_dir, lastDir, f),
                        )

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)  # type: ignore

    # start training
    runner.train()


if __name__ == "__main__":
    main()
