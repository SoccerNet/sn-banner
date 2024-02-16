_base_ = [
    "pointrend_r50.py",
    "../soccernet.py",
    "../default_runtime.py",
    "../schedule_40k.py",
]
crop_size = (1080, 1920)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
param_scheduler = [
    dict(type="LinearLR", by_epoch=False, start_factor=0.1, begin=0, end=200 // 2),
    dict(
        type="PolyLR",
        eta_min=1e-4,
        power=0.9,
        begin=200 // 2,
        end=80000 // 2,
        by_epoch=False,
    ),
]
