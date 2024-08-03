# Credits:
#
#   -   https://github.com/mguti97/No-Bells-Just-Whistles/blob/main/scripts/inference_sn.py

import os
import sys
import json
import glob
import yaml
import torch
import zipfile
import argparse
import warnings
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from No_Bells_Just_Whistles.model.cls_hrnet import get_cls_net
from No_Bells_Just_Whistles.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from No_Bells_Just_Whistles.utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)
from No_Bells_Just_Whistles.utils.utils_calib import FramebyFrameCalib
from No_Bells_Just_Whistles.utils.utils_calib2 import (
    FramebyFrameCalib as FramebyFrameCalib2,
)
from No_Bells_Just_Whistles.utils.utils_calib3 import (
    FramebyFrameCalib as FramebyFrameCalib3,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def per_dir_inference(
    source_dir,
    inference_dir,
    args,
    cam,
    model,
    model_l,
    transform,
    gt_zip_name,
    pred_zip_name,
):
    files = glob.glob(os.path.join(source_dir, "*.jpg"))
    files.sort()  #! NOT Optional

    zip_name_gt = os.path.join(inference_dir, gt_zip_name)

    zip_name_pred = os.path.join(inference_dir, pred_zip_name)

    with zipfile.ZipFile(zip_name_pred, "w") as zip_file:
        directory_name = source_dir.split("/")[-2]
        tqdm_desc = f"Processing Images from {directory_name}"
        for file in tqdm(files, desc=tqdm_desc, leave=False, disable=True):
            image = Image.open(file)
            file_name = file.split("/")[-1].split(".")[0]

            with torch.no_grad():
                image = f.to_tensor(image).float().to(device).unsqueeze(0)
                image = image if image.size()[-1] == 960 else transform(image)
                _, _, h, w = image.size()
                heatmaps = model(image)
                heatmaps_l = model_l(image)

                kp_coords = get_keypoints_from_heatmap_batch_maxpool(
                    heatmaps[:, :-1, :, :]
                )
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(
                    heatmaps_l[:, :-1, :, :]
                )
                kp_dict = coords_to_dict(kp_coords, threshold=args.kp_th)
                lines_dict = coords_to_dict(line_coords, threshold=args.line_th)
                final_dict = complete_keypoints(
                    kp_dict, lines_dict, w=w, h=h, normalize=True
                )

                cam.update(final_dict[0])
                final_params_dict = cam.heuristic_voting()

            if final_params_dict:
                if final_params_dict["rep_err"] <= args.max_reproj_err:
                    cam_params = final_params_dict["cam_params"]
                    json_data = json.dumps(cam_params)
                    zip_file.writestr(f"camera_{file_name}.json", json_data)

    labelsGamestateJson = json.load(
        open(os.path.join(source_dir, "../Labels-GameState.json"))
    )["annotations"]
    pitch_gt_annotations = [
        x["lines"] for x in labelsGamestateJson if x["supercategory"] == "pitch"
    ]
    with zipfile.ZipFile(zip_name_gt, "w") as zip_file:
        for i, file in enumerate(files):
            file_name = file.split("/")[-1].split(".")[0]
            data = pitch_gt_annotations[i]
            json_data = json.dumps(data)
            zip_file.writestr(f"{file_name}.json", json_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the (kp model) configuration file",
    )
    parser.add_argument(
        "--cfg_l",
        type=str,
        required=True,
        help="Path to the (line model) configuration file",
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")
    parser.add_argument("--save_dir", type=str, required=True, help="Saving file path")
    parser.add_argument(
        "--weights_kp", type=str, required=True, help="Model (keypoints) weigths to use"
    )
    parser.add_argument(
        "--weights_line", type=str, required=True, help="Model (lines) weigths to use"
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="cuda:0",
        help="CUDA device index (default: 'cuda:0')",
    )
    parser.add_argument(
        "--kp_th", type=float, default=0.1486, help="Keypoint threshold"
    )
    parser.add_argument("--line_th", type=float, default=0.388, help="Line threshold")
    parser.add_argument("--max_reproj_err", type=float, default="25")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--array_index", type=int, default=-1)
    parser.add_argument(
        "-v", "--version", type=int, required=True, help="Version of NBJW"
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
    parser.add_argument("-t", "--test", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    width = args.width
    height = args.height

    directories = os.listdir(os.path.join(args.root_dir, args.split))
    # remove "sequences_info.json" from the list
    directories = [x for x in directories if x != "sequences_info.json"]
    directories.sort()
    # split in 4

    if args.array_index != -1:
        directories = np.array_split(directories, 4)[args.array_index]

    if args.test:
        directories = directories[:2]

    source_dirs = [
        os.path.join(args.root_dir, args.split, x, "img1") for x in directories
    ]

    inference_dirs = [os.path.join(args.save_dir, args.split, x) for x in directories]
    # create inference directories
    for directory in inference_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Preprocessing (loading model, etc.)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.cfg, "r"))
    cfg_l = yaml.safe_load(open(args.cfg_l, "r"))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform = T.Resize((540, 960))
    if args.version == 1:
        cam = FramebyFrameCalib(width, height, denormalize=True)
    elif args.version == 2:
        cam = FramebyFrameCalib2(width, height, denormalize=True)
    elif args.version == 3:
        cam = FramebyFrameCalib3(width, height, denormalize=True)
    else:
        raise ValueError("Invalid version number, must be 1, 2 or 3")

    for source_dir, inference_dir in tqdm(
        zip(source_dirs, inference_dirs),
        desc="Processing Directories",
        total=len(directories),
    ):
        per_dir_inference(
            source_dir,
            inference_dir,
            args,
            cam,
            model,
            model_l,
            transform,
            args.gt_zip_name,
            args.pred_zip_name,
        )
