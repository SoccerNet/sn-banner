# Config for test purposes

_base_ = [
    "configs/_base_/models/pspnet_r50-d8.py",
    "configs/_base_/default_runtime.py",
    "soccernet.py",
    "configs/_base_/schedules/schedule_40k.py",
]

norm_cfg = dict(type="BN", requires_grad=True)
crop_size_height = int(9 * 100)
crop_size = (crop_size_height, crop_size_height * 16 // 9)
scale = (crop_size[1], crop_size[0])
data_preprocessor = dict(size=crop_size)
num_classes = 3
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg, num_classes=num_classes),
    auxiliary_head=dict(norm_cfg=norm_cfg, num_classes=num_classes),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(
    #     type="RandomResize", scale=(1920, 1080), ratio_range=(0.5, 2.0), keep_ratio=True
    # ),
    # dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

val_interval = 50
train_cfg = dict(val_interval=val_interval)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        save_best="mIoU",
        interval=val_interval,
        max_keep_ckpts=2,
    ),
)

# We can use the pre-trained model to obtain higher performance
# load_from = "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_094027-2a90a4a3.pth"

# Set random seed to improve reproducibility
seed = 0
# randomness = dict(seed=seed, deterministic=True)
randomness = dict(seed=seed)

# default_hooks = dict(visualization=dict(draw=True), logger=dict(interval=10))
# default_hooks = dict(logger=dict(interval=10))
