# GPU memory size: 20.450GB

_base_ = "knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512.py"

checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220308-f41b89d3.pth"  # noqa

# model settings
norm_cfg = dict(type="BN", requires_grad=True)
num_stages = 3
conv_kernel_size = 1

model = dict(
    type="EncoderDecoder",
    pretrained=checkpoint_file,
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(kernel_generate_head=dict(in_channels=[96, 192, 384, 768])),
    auxiliary_head=dict(in_channels=384),
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    # modify learning rate following the official implementation of Swin Transformer # noqa
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000 // 2),
    dict(
        type="MultiStepLR",
        begin=1000 // 2,
        end=80000 // 2,
        milestones=[60000 // 2, 72000 // 2],
        by_epoch=False,
    ),
]
