ann_dir = 'Labels'
crop_size = (
    1080,
    1920,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1080,
        1920,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '../Dataset'
dataset_type = 'SoccerNet'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=135,
        max_keep_ckpts=2,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=135, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_dir = 'Images'
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
]
iters = 40000
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        contract_dilation=True,
        depth=101,
        dilations=(
            1,
            1,
            1,
            1,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            2,
            2,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1080,
            1920,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=[
        dict(
            align_corners=False,
            channels=128,
            dropout_ratio=-1,
            feature_strides=[
                4,
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                256,
                256,
                256,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_classes=4,
            type='FPNHead'),
        dict(
            align_corners=False,
            channels=256,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            in_channels=[
                256,
            ],
            in_index=[
                0,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=4,
            num_fcs=3,
            type='PointHead'),
    ],
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    num_stages=2,
    pretrained='open-mmlab://resnet101_v1c',
    test_cfg=dict(
        mode='whole',
        scale_factor=2,
        subdivision_num_points=8196,
        subdivision_steps=2),
    train_cfg=dict(
        importance_sample_ratio=0.75, num_points=2048, oversample_ratio=3),
    type='CascadeEncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
num_classes = 4
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=100, start_factor=0.1, type='LinearLR'),
    dict(
        begin=100,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
scale = (
    1920,
    1080,
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='splits/test.txt',
        data_prefix=dict(img_path='Images', seg_map_path='Labels'),
        data_root='../Dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1920,
                1080,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SoccerNet'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=135)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='Images', seg_map_path='Labels'),
        data_root='../Dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1920,
                    1080,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    1080,
                    1920,
                ),
                type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='SoccerNet'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1920,
            1080,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        1080,
        1920,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=2.0, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='Images', seg_map_path='Labels'),
        data_root='../Dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1920,
                1080,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='SoccerNet'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
val_interval = 135
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1920,
        1080,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/pointrend_r101_4xb2-80k_cityscapes-512x1024'
