_base_ = [
    "../soccernet.py",
    "../default_runtime.py",
]

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23-in1kpre_3rdparty-9ca29f62.pth"  # noqa
crop_size = (1080, 1920)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
norm_cfg = dict(type="SyncBN", requires_grad=True)
num_classes = 3
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DDRNet",
        in_channels=3,
        channels=64,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
    ),
    decode_head=dict(
        type="DDRHead",
        in_channels=64 * 4,
        channels=128,
        dropout_ratio=0.0,
        num_classes=num_classes,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=[1.0] * num_classes,
                loss_weight=1.0,
            ),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=[1.0] * num_classes,
                loss_weight=0.4,
            ),
        ],
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_dataloader = dict(batch_size=6, num_workers=4)

iters = 40000
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=iters, by_epoch=False)
]

# training schedule for 40k
val_interval = 100
train_cfg = dict(type="IterBasedTrainLoop", max_iters=iters, val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        save_best="mIoU",
        interval=val_interval,
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

# randomness = dict(seed=304)
