import sys
sys.path.append("../../")

from contrib.pretrain.datasets import PretrainDataset
from prefusion.registry import DATASETS
from prefusion.runner import GroupRunner


dataset_type = 'prefusion.PretrainDataset'
data_root = 'data/pretrain_data/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='prefusion.LoadAnnotationsPretrain', with_bbox=False, with_label=False, with_seg=True, with_depth=True, with_seg_mask=True),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 1024),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="mv_4d_infos_train.pkl",
        pipeline=train_pipeline,
        camera_types=["VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_FRONT", "VCAMERA_FISHEYE_LEFT", "VCAMERA_FISHEYE_RIGHT"])
    )

dataloader = GroupRunner.build_dataloader(train_dataloader)

for data in dataloader:
    print(data)