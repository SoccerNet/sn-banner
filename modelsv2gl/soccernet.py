# dataset settings
dataset_type = "SoccerNetv2Gl"
data_root = "Datasetv2"
img_dir = "Images"
ann_dir = "Goal/Labels"
# i = 50
# crop_size = (1080 - i, 1920 - i)
# scale = (1920 - i, 1080 - i)
crop_size = (1080, 1920)
scale = (1920, 1080)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(
    #     type="RandomResize",
    #     scale=scale,
    #     ratio_range=(
    #         1.0,
    #         3.0,
    #     ),  # Maybe no scale down, to avoid losing details, so keep ratio_range=(1.0, a) where a > 1.0
    #     keep_ratio=True,
    # ),
    # dict(type="CustomTransform", crop_size=crop_size, cat_max_ratio=0.90),
    dict(type="RandomFlip", prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=train_pipeline,
        ann_file="Goal/splits/train.txt",
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=val_pipeline,
        ann_file="Goal/splits/val.txt",
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=val_pipeline,
        ann_file="Goal/splits/test.txt",
    ),
)

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
