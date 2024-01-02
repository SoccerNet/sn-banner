_base_ = [
    "configs/_base_/models/pspnet_r50-d8.py",
    "soccernetv2.py",
    "configs/_base_/default_runtime.py",
    "configs/_base_/schedules/schedule_80k.py",
]

crop_size = (1080, 1920)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type="BN", requires_grad=True)
num_classes = 3
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(norm_cfg=norm_cfg, depth=101),
    decode_head=dict(norm_cfg=norm_cfg, num_classes=num_classes),
    auxiliary_head=dict(norm_cfg=norm_cfg, num_classes=num_classes),
    pretrained="open-mmlab://resnet101_v1c",
)

# We can use the pre-trained model to obtain higher performance
# load_from = "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_094027-2a90a4a3.pth"

# default_hooks = dict(visualization=dict(draw=True), logger=dict(interval=10))
# default_hooks = dict(logger=dict(interval=10))
