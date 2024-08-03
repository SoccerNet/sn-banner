ann_dir = 'Labels'
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220308-f41b89d3.pth'
conv_kernel_size = 1
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
    logger=dict(interval=15, log_metric_by_epoch=False, type='LoggerHook'),
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
load_from = 'work_dirs/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512/best_mIoU_iter_10665.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=384,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=4,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        attn_drop_rate=0.0,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=96,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        use_abs_pos_embed=False,
        window_size=7),
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
    decode_head=dict(
        kernel_generate_head=dict(
            align_corners=False,
            channels=512,
            dropout_ratio=0.1,
            in_channels=[
                96,
                192,
                384,
                768,
            ],
            in_index=[
                0,
                1,
                2,
                3,
            ],
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            num_classes=4,
            pool_scales=(
                1,
                2,
                3,
                6,
            ),
            type='UPerHead'),
        kernel_update_head=[
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=4,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=4,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
            dict(
                conv_kernel_size=1,
                dropout=0.0,
                feat_transform_cfg=dict(
                    act_cfg=None, conv_cfg=dict(type='Conv2d')),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=512,
                kernel_updator_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=256,
                    in_channels=256,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='KernelUpdator'),
                num_classes=4,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                out_channels=512,
                type='KernelUpdateHead',
                with_ffn=True),
        ],
        num_stages=3,
        type='IterativeDecodeHead'),
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220308-f41b89d3.pth',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 4
num_stages = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=500,
        by_epoch=False,
        end=40000,
        milestones=[
            30000,
            36000,
        ],
        type='MultiStepLR'),
]
resume = False
scale = (
    1920,
    1080,
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
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
    ],
    keep_results=True,
    output_dir='inferences/knet/test/non-tta',
    type='IoUMetric')
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
    batch_size=2,
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
work_dir = './work_dirs/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512/test/non-tta'
