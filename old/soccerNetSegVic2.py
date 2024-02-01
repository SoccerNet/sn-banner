import torch, mmseg, mmcv, mmengine, os, shutil, numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

# define dataset root and directory for images and annotations
data_root = "Dataset"
img_dir = "Images"
ann_dir = "Labels"

print("ok1")

# Color code of the annotation images
#                              R-G-B
# ---------------------------------------------------------
# Outside billboards

#     0.                    000-000-000  (black)

# ---------------------------------------------------------
# Inside billboards

#     1. Billboard          255-255-255  (white)

#     2. Field player       255-000-000  (red)
#     3. Goalkeeper         000-255-000  (green)
#     4. Referee            000-000-255  (blue)
#     5. Assistant referee  255-255-000  (yellow)
#     6. Other human        255-000-255  (pink)

#     7. Ball               000-255-255  (turquoise)

#     8. Goal post          128-000-000  (dark red)
#     9. Goal net           000-128-000  (dark green)
#    10. Net post           000-000-128  (dark blue)
#    11. Cross-bar          064-064-064  (dark gray)

#    12. Corner flag        128-128-000  (dark yellow)
#    13. Assistant flag     128-000-128  (purple)

#    14. Microphone         000-128-128  (dark turquoise)
#    15. Camera             255-128-000  (orange)

#    16. Other object       192-192-192  (light gray)

#    17. Don't care         128-128-128  (gray)
# ---------------------------------------------------------

# Define the classes and the palette for the segmentation of my dataset
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
old_palette_dict = {tuple(p): i for i, p in enumerate(old_palette)}
old_classes_dict = {i: c for i, c in enumerate(old_classes)}

new_classes = ["Outside billboards", "Billboard", "Goal net"]
# new_classes = old_classes
# new_classes = ['Outside billboards', 'Goal net']

new_palette = [old_palette[old_classes.index(c)] for c in new_classes]
new_palette_dict = {tuple(p): i for i, p in enumerate(new_palette)}
new_classes_dict = {i: c for i, c in enumerate(new_classes)}

print("ok2")

conversion_old_to_new = np.zeros(
    len(old_classes), dtype=np.uint8
)  # By default, all classes are mapped to 0 (Outside billboards)
# conversion_old_to_new = np.array([2 for i in range(len(old_classes))], dtype=np.uint8)  # By default, all classes are mapped to 2 (Other object)
conversion_old_to_new[old_classes.index("Billboard")] = new_classes.index("Billboard")
conversion_old_to_new[old_classes.index("Goal net")] = new_classes.index("Goal net")

# conversion_old_to_new = np.array([i for i in range(len(old_classes))], dtype=np.uint8)  # No conversion

# conversion_old_to_new[old_classes.index('Goal net')] = 1  # Goal net
print(conversion_old_to_new)

ann_source_dir = "Labels_source"

all_ann_files = os.listdir(os.path.join(data_root, ann_source_dir))

print("ok3")


def convert_ann_file(files):
    for file in tqdm(files):
        # Open the image
        ann = Image.open(os.path.join(data_root, ann_source_dir, file))
        # Create a numpy array from the image
        ann_np = np.array(ann, dtype=np.uint8)
        # For each pixel x, replace it by the value conversion_old_to_new[x]
        ann_np = conversion_old_to_new[ann_np]
        # Create the image from the numpy array
        seg_img = Image.fromarray(ann_np).convert("P")
        # Add the new palette to the image
        seg_img.putpalette(np.array(new_palette, dtype=np.uint8))
        # Save the image
        seg_img.save(os.path.join(data_root, ann_dir, file))


print("ok4")

# convert_ann_file(all_ann_files)

print("ok5")

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class SoccerNet(BaseSegDataset):
    METAINFO = dict(classes=new_classes, palette=new_palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".png", seg_map_suffix=".png", **kwargs)


from mmengine import Config

cfg = Config.fromfile("pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py")

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = len(new_classes)
cfg.model.auxiliary_head.num_classes = len(new_classes)

# Modify dataset type and path
cfg.dataset_type = "SoccerNet"
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 2

cfg.train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    # dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]

cfg.test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]


cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = "splits/train.txt"

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = "splits/val.txt"

cfg.test_dataloader = cfg.val_dataloader


# Load the pretrained weights
cfg.load_from = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = "./work_dirs/soccerNetSegVic2"

cfg.train_cfg.max_iters = 40000
cfg.train_cfg.val_interval = 200
cfg.default_hooks.logger.interval = 50
cfg.default_hooks.checkpoint.interval = 200

# Set seed to facilitate reproducing the result
cfg["randomness"] = dict(seed=0)

# Let's have a look at the final config used for training
print(f"Config:\n{cfg.pretty_text}")

from mmengine.runner import Runner

runner = Runner.from_cfg(cfg)

print("beginning training")

# start training
runner.train()
