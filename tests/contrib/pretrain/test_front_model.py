image_size = (1024, 1024)
batch_augments = [dict(type='mmdet.BatchFixedSizePad', size=image_size)]

data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[128, 128, 128],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments)


model_config = dict(
    type='mmdet.FCOS',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='VoVNet',
                  out_indices=(0, 1, 3, 5),
                  ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=4,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='mmdet.FCOSHead',
        num_classes=37,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)

import torch
from prefusion.registry import MODELS, DATASETS
from contrib.pretrain.models.vovnet import VoVNet
from pathlib import Path
from mmengine.dataset import Compose
from contrib.pretrain.datasets.dataset import PretrainDataset_FrontData


def test_front_model():
    # Create a dummy input tensor
    inputs = torch.randn(1, 3, 1024, 1024)
    data_samples = [dict(img_path='dummy_path')]
    
    # Initialize the model
    backbone = MODELS.build(model_config.get('backbone'))  # type: ignore
    neck = MODELS.build(model_config.get('neck'))  # type: ignore 
    head = MODELS.build(model_config.get('bbox_head'))  # type: ignore
    # Forward pass
    with torch.no_grad():
        backbone_output = backbone(inputs)
        neck_output = neck(backbone_output)
        head_output = head(neck_output)
        # model = MODELS.build(model_config)
        # out = model(inputs, data_samples, mode='loss')
    
    assert len(backbone_output) == 4, "backbone output should have 4 feature maps"
    assert len(neck_output) == 4, "neck output should have 4 feature maps"
    assert len(head_output) == 3, "head output should have 3 feature maps"
    