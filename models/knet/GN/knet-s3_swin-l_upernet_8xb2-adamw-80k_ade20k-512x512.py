# GPU memory size: 22.387GB

_base_ = "knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py"

checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth"  # noqa
# model settings
model = dict(
    pretrained=checkpoint_file,
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(kernel_generate_head=dict(in_channels=[192, 384, 768, 1536])),
    auxiliary_head=dict(in_channels=768),
)
