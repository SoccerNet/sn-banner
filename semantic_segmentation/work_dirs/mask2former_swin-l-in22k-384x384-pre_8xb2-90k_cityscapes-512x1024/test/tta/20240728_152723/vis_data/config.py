ann_dir = 'Labels'
auto_scale_lr = dict(base_batch_size=2, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
crop_size = (
    1080,
    1920,
)
custom_keys = dict({
    'absolute_pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.10.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=0.1)
})
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
    test_cfg=dict(size_divisor=32),
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
    visualization=dict(draw=True, interval=100, type='SegVisualizationHook'))
default_scope = 'mmseg'
depths = [
    2,
    2,
    18,
    2,
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
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
load_from = 'work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024/20240725_222653/best_mIoU_iter_10935.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    module=dict(
        backbone=dict(
            attn_drop_rate=0.0,
            depths=[
                2,
                2,
                18,
                2,
            ],
            drop_path_rate=0.3,
            drop_rate=0.0,
            embed_dims=192,
            frozen_stages=-1,
            init_cfg=dict(
                checkpoint=
                'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth',
                type='Pretrained'),
            mlp_ratio=4,
            num_heads=[
                6,
                12,
                24,
                48,
            ],
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            patch_norm=True,
            pretrain_img_size=384,
            qk_scale=None,
            qkv_bias=True,
            type='SwinTransformer',
            window_size=12,
            with_cp=False),
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
            test_cfg=dict(size_divisor=32),
            type='SegDataPreProcessor'),
        decode_head=dict(
            align_corners=False,
            enforce_decoder_input_project=False,
            feat_channels=256,
            in_channels=[
                192,
                384,
                768,
                1536,
            ],
            loss_cls=dict(
                class_weight=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.1,
                ],
                loss_weight=2.0,
                reduction='mean',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False),
            loss_dice=dict(
                activate=True,
                eps=1.0,
                loss_weight=5.0,
                naive_dice=True,
                reduction='mean',
                type='mmdet.DiceLoss',
                use_sigmoid=True),
            loss_mask=dict(
                loss_weight=5.0,
                reduction='mean',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            num_classes=4,
            num_queries=100,
            num_transformer_feat_level=3,
            out_channels=256,
            pixel_decoder=dict(
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    init_cfg=None,
                    layer_cfg=dict(
                        ffn_cfg=dict(
                            act_cfg=dict(inplace=True, type='ReLU'),
                            embed_dims=256,
                            feedforward_channels=1024,
                            ffn_drop=0.0,
                            num_fcs=2),
                        self_attn_cfg=dict(
                            batch_first=True,
                            dropout=0.0,
                            embed_dims=256,
                            im2col_step=64,
                            init_cfg=None,
                            norm_cfg=None,
                            num_heads=8,
                            num_levels=3,
                            num_points=4)),
                    num_layers=6),
                init_cfg=None,
                norm_cfg=dict(num_groups=32, type='GN'),
                num_outs=3,
                positional_encoding=dict(normalize=True, num_feats=128),
                type='mmdet.MSDeformAttnPixelDecoder'),
            positional_encoding=dict(normalize=True, num_feats=128),
            strides=[
                4,
                8,
                16,
                32,
            ],
            train_cfg=dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(
                            type='mmdet.CrossEntropyLossCost',
                            use_sigmoid=True,
                            weight=5.0),
                        dict(
                            eps=1.0,
                            pred_act=True,
                            type='mmdet.DiceCost',
                            weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                importance_sample_ratio=0.75,
                num_points=12544,
                oversample_ratio=3.0,
                sampler=dict(type='mmdet.MaskPseudoSampler')),
            transformer_decoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    cross_attn_cfg=dict(
                        attn_drop=0.0,
                        batch_first=True,
                        dropout_layer=None,
                        embed_dims=256,
                        num_heads=8,
                        proj_drop=0.0),
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        add_identity=True,
                        dropout_layer=None,
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        attn_drop=0.0,
                        batch_first=True,
                        dropout_layer=None,
                        embed_dims=256,
                        num_heads=8,
                        proj_drop=0.0)),
                num_layers=9,
                return_intermediate=True),
            type='Mask2FormerHead'),
        test_cfg=dict(mode='whole'),
        train_cfg=dict(),
        type='EncoderDecoder'),
    type='SegTTAModel')
num_classes = 4
optim_wrapper = dict(
    accumulative_counts=2,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.0001,
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
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
    output_dir='inferences/test/mask2former/tta',
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
tta_model = dict(
    module=dict(
        backbone=dict(
            attn_drop_rate=0.0,
            depths=[
                2,
                2,
                18,
                2,
            ],
            drop_path_rate=0.3,
            drop_rate=0.0,
            embed_dims=192,
            frozen_stages=-1,
            init_cfg=dict(
                checkpoint=
                'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth',
                type='Pretrained'),
            mlp_ratio=4,
            num_heads=[
                6,
                12,
                24,
                48,
            ],
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            patch_norm=True,
            pretrain_img_size=384,
            qk_scale=None,
            qkv_bias=True,
            type='SwinTransformer',
            window_size=12,
            with_cp=False),
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
            test_cfg=dict(size_divisor=32),
            type='SegDataPreProcessor'),
        decode_head=dict(
            align_corners=False,
            enforce_decoder_input_project=False,
            feat_channels=256,
            in_channels=[
                192,
                384,
                768,
                1536,
            ],
            loss_cls=dict(
                class_weight=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.1,
                ],
                loss_weight=2.0,
                reduction='mean',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False),
            loss_dice=dict(
                activate=True,
                eps=1.0,
                loss_weight=5.0,
                naive_dice=True,
                reduction='mean',
                type='mmdet.DiceLoss',
                use_sigmoid=True),
            loss_mask=dict(
                loss_weight=5.0,
                reduction='mean',
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True),
            num_classes=4,
            num_queries=100,
            num_transformer_feat_level=3,
            out_channels=256,
            pixel_decoder=dict(
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    init_cfg=None,
                    layer_cfg=dict(
                        ffn_cfg=dict(
                            act_cfg=dict(inplace=True, type='ReLU'),
                            embed_dims=256,
                            feedforward_channels=1024,
                            ffn_drop=0.0,
                            num_fcs=2),
                        self_attn_cfg=dict(
                            batch_first=True,
                            dropout=0.0,
                            embed_dims=256,
                            im2col_step=64,
                            init_cfg=None,
                            norm_cfg=None,
                            num_heads=8,
                            num_levels=3,
                            num_points=4)),
                    num_layers=6),
                init_cfg=None,
                norm_cfg=dict(num_groups=32, type='GN'),
                num_outs=3,
                positional_encoding=dict(normalize=True, num_feats=128),
                type='mmdet.MSDeformAttnPixelDecoder'),
            positional_encoding=dict(normalize=True, num_feats=128),
            strides=[
                4,
                8,
                16,
                32,
            ],
            train_cfg=dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.ClassificationCost', weight=2.0),
                        dict(
                            type='mmdet.CrossEntropyLossCost',
                            use_sigmoid=True,
                            weight=5.0),
                        dict(
                            eps=1.0,
                            pred_act=True,
                            type='mmdet.DiceCost',
                            weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                importance_sample_ratio=0.75,
                num_points=12544,
                oversample_ratio=3.0,
                sampler=dict(type='mmdet.MaskPseudoSampler')),
            transformer_decoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    cross_attn_cfg=dict(
                        attn_drop=0.0,
                        batch_first=True,
                        dropout_layer=None,
                        embed_dims=256,
                        num_heads=8,
                        proj_drop=0.0),
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        add_identity=True,
                        dropout_layer=None,
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        attn_drop=0.0,
                        batch_first=True,
                        dropout_layer=None,
                        embed_dims=256,
                        num_heads=8,
                        proj_drop=0.0)),
                num_layers=9,
                return_intermediate=True),
            type='Mask2FormerHead'),
        test_cfg=dict(mode='whole'),
        train_cfg=dict(),
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
work_dir = './work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024/test/tta'
