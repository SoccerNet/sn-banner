iters = 40000

# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=iters, by_epoch=False)
]
# training schedule for 40k
val_interval = 150
train_cfg = dict(type="IterBasedTrainLoop", max_iters=iters, val_interval=val_interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=75, log_metric_by_epoch=False),
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
