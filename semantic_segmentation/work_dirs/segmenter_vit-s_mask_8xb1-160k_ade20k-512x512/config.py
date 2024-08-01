ann_dir = 'Labels'
backbone_norm_cfg = dict(eps=1e-06, requires_grad=True, type='LN')
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'
crop_size = (
    1080,
    1920,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1080,
        1920,
    ),
    std=[
        127.5,
        127.5,
        127.5,
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
load_from = 'work_dirs/segmenter_vit-s_mask_8xb1-160k_ade20k-512x512/best_mIoU_iter_30375.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    module=dict(
        backbone=dict(
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            drop_rate=0.0,
            embed_dims=384,
            final_norm=True,
            img_size=(
                1080,
                1920,
            ),
            in_channels=3,
            interpolate_mode='bicubic',
            norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
            num_heads=6,
            num_layers=12,
            patch_size=16,
            type='VisionTransformer',
            with_cls_token=True),
        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            pad_val=0,
            seg_pad_val=255,
            size=(
                1080,
                1920,
            ),
            std=[
                127.5,
                127.5,
                127.5,
            ],
            type='SegDataPreProcessor'),
        decode_head=dict(
            channels=384,
            dropout_ratio=0.0,
            embed_dims=384,
            in_channels=384,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=150,
            num_heads=6,
            num_layers=2,
            type='SegmenterMaskTransformerHead'),
        pretrained=
        'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth',
        test_cfg=dict(
            crop_size=(
                512,
                512,
            ), mode='slide', stride=(
                480,
                480,
            )),
        type='EncoderDecoder'),
    type='SegTTAModel')
num_classes = 4
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0),
    type='OptimWrapper')
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0)
param_scheduler = [
    dict(
        begin=0,
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
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                transforms=[
                    [
                        dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                        dict(
                            keep_ratio=True, scale_factor=0.75, type='Resize'),
                        dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                        dict(
                            keep_ratio=True, scale_factor=1.25, type='Resize'),
                        dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                        dict(
                            keep_ratio=True, scale_factor=1.75, type='Resize'),
                        dict(keep_ratio=True, scale_factor=2.0, type='Resize'),
                    ],
                    [
                        dict(
                            direction='horizontal',
                            prob=0.0,
                            type='RandomFlip'),
                        dict(
                            direction='horizontal',
                            prob=1.0,
                            type='RandomFlip'),
                    ],
                    [
                        dict(type='LoadAnnotations'),
                    ],
                    [
                        dict(type='PackSegInputs'),
                    ],
                ],
                type='TestTimeAug'),
        ],
        type='SoccerNet'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    keep_results=True,
    output_dir='inferences/segmenter/tta',
    type='IoUMetric')
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=135)
train_dataloader = dict(
    batch_size=1,
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
    num_workers=2,
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
tta_model = dict(
    module=dict(
        backbone=dict(
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            drop_rate=0.0,
            embed_dims=384,
            final_norm=True,
            img_size=(
                1080,
                1920,
            ),
            in_channels=3,
            interpolate_mode='bicubic',
            norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
            num_heads=6,
            num_layers=12,
            patch_size=16,
            type='VisionTransformer',
            with_cls_token=True),
        data_preprocessor=dict(
            bgr_to_rgb=True,
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            pad_val=0,
            seg_pad_val=255,
            size=(
                1080,
                1920,
            ),
            std=[
                127.5,
                127.5,
                127.5,
            ],
            type='SegDataPreProcessor'),
        decode_head=dict(
            channels=384,
            dropout_ratio=0.0,
            embed_dims=384,
            in_channels=384,
            loss_decode=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=150,
            num_heads=6,
            num_layers=2,
            type='SegmenterMaskTransformerHead'),
        pretrained=
        'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth',
        test_cfg=dict(
            crop_size=(
                512,
                512,
            ), mode='slide', stride=(
                480,
                480,
            )),
        type='EncoderDecoder'),
    type='SegTTAModel')
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
    num_workers=2,
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
work_dir = './work_dirs/segmenter_vit-s_mask_8xb1-160k_ade20k-512x512'
