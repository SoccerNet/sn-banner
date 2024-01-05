from mmseg.apis import MMSegInferencer

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

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
    model="mask2former_swin-t_1xb2-90k_soccernet.py",
    weights="best_mIoU_iter_46200.pth",
    classes=new_classes,
    palette=new_palette,
)

inferencer("Dataset/Images/Stadium_4_Match_2_in022.png", show=True)
