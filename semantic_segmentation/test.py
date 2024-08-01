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


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
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
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="Inference on video")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--challenge", action="store_true", help="Challenge split evaluation"
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if "visualization" in default_hooks:
        visualization_hook = default_hooks["visualization"]
        # Turn on visualization
        visualization_hook["draw"] = True
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer["save_dir"] = args.show_dir
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    print("cfg.work_dir", cfg.work_dir)
    cfg.launcher = args.launcher
    cfg.default_hooks.logger.interval = 1
    if args.video:
        cfg.test_dataloader.dataset.ann_file = "test.txt"
        cfg.test_dataloader.dataset.type = "Video3"
        cfg.test_dataloader.dataset.data_root = "video"
        cfg.test_dataloader.dataset.data_prefix.img_path = "Images/360"
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print("args.work_dir", args.work_dir)
    print("cfg.work_dir", cfg.work_dir)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            cfg.work_dir,
            "challenge" if args.challenge else "test",
            "tta" if args.tta else "non-tta",
        )
        print("new cfg.work_dir", cfg.work_dir)

    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
        cfg.test_dataloader.num_workers = args.batch_size

    if args.challenge:
        cfg.test_dataloader.dataset.ann_file = "splits/challenge.txt"

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        # If tta is enabled, add the directory 'tta/' to out, else add 'non-tta/'
        if args.tta:
            args.out = osp.join(args.out, "tta")
        else:
            args.out = osp.join(args.out, "non-tta")
        cfg.test_evaluator["output_dir"] = args.out
        cfg.test_evaluator["keep_results"] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == "__main__":
    main()
