# GPU memory size: 17.283GB

_base_ = ["./segformer_mit-b0_8xb2-160k_ade20k-512x512.py"]

val_dataloader = dict(batch_size=8)

checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth"  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 8, 27, 3],
    ),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
)
