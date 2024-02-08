from mmseg.apis import init_model, inference_model
from mmengine.config import Config
import torch
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE
import numpy as np

from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils.typing_utils import SampleList

from mmseg.apis import MMSegInferencer

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

from PIL import Image

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


inferencer = MMSegInferencer(
    model="./models/mask2former/mask2former_swin-t_1xb2-90k_soccernet.py",
    weights="best_mIoU_iter_46200.pth",
    classes=new_classes,
    palette=new_palette,
)

img = "./Dataset/Images/Stadium_1_Match_1_in_1fps_0491.png"

result = inferencer(img)

from tqdm import tqdm


pred = torch.tensor(result["predictions"], device="cuda", dtype=torch.long)

gt = torch.tensor(
    np.array(Image.open("./Dataset/Labels/Stadium_1_Match_1_in_1fps_0491.png")),
    device="cuda",
    dtype=torch.long,
)

nclasses = 18

gt_one_hot = (
    torch.nn.functional.one_hot(gt, num_classes=nclasses).permute(2, 0, 1).float()
)
pred_one_hot = (
    torch.nn.functional.one_hot(pred, num_classes=nclasses).permute(2, 0, 1).float()
)

kernel = (
    torch.tensor(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        dtype=torch.float32,
        device=pred.device,
    )
    .unsqueeze(0)
    .unsqueeze(0)
)

pred_borders = (
    torch.nn.functional.conv2d(pred_one_hot.unsqueeze(1), kernel, padding="same")
    .squeeze()
    .clamp(0, 1)
)
gt_borders = (
    torch.nn.functional.conv2d(gt_one_hot.unsqueeze(1), kernel, padding="same")
    .squeeze()
    .clamp(0, 1)
)

d = 3

kernel = np.ones((d * 2 + 1, d * 2 + 1), dtype=np.uint8)
kernel[d, d] = 0
kernel = (
    torch.tensor(
        distanceTransform(kernel, DIST_L2, DIST_MASK_PRECISE),
        dtype=torch.float32,
        device="cuda",
    )
    .le(d)
    .float()
    .unsqueeze(0)
    .unsqueeze(0)
)
pred_d_border = (
    torch.nn.functional.conv2d(pred_borders.unsqueeze(1), kernel, padding="same")
    .squeeze()
    .clamp(0, 1)
)
gt_d_border = (
    torch.nn.functional.conv2d(gt_borders.unsqueeze(1), kernel, padding="same")
    .squeeze()
    .clamp(0, 1)
)
pred_intersections = torch.logical_and(pred_one_hot, pred_d_border)
gt_intersections = torch.logical_and(gt_one_hot, gt_d_border)
# print(pred_intersections[2].sum(), gt_intersections[2].sum())
