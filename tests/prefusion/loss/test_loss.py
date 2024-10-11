import functools

import pytest
import torch
import numpy as np

from prefusion.loss.basic import SegIouLoss, seg_iou

_approx = functools.partial(pytest.approx, rel=1e-5)

@pytest.fixture
def seg_pred():
    return torch.tensor([[
        [[0.0, 0.1,  0.2, 0.3, 0.1],
         [0.0, 0.2,  0.8, 0.1, 0.1],
         [0.0, 0.15, 0.5, 0.7, 0.2],
         [0.1, 0.15, 0.1, 0.0, 0.0],],
         
        [[0.7, 0.2, 0.0, 1.0, 0.5],
         [0.1, 0.0, 0.0, 0.0, 0.2],
         [0.1, 0.0, 0.0, 0.0, 0.3],
         [0.9, 0.1, 0.0, 0.2, 0.8],],
    ]])


@pytest.fixture
def seg_label():
    return torch.tensor([[
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0],],
         
        [[1, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],],
    ]])


def test_seg_iou(seg_pred, seg_label):
    assert seg_iou(seg_pred[0, 0], seg_label[0, 0]).item() == _approx(0.51724)
    assert seg_iou(seg_pred[0, 1], seg_label[0, 1]).item() == _approx(0.541666)
    assert seg_iou(seg_pred, seg_label, dim=(0, 2, 3)).tolist() == _approx([0.51724, 0.541666])
    assert seg_iou(seg_pred[0, 1], torch.zeros_like(seg_pred[0, 1])).item() == _approx(0.75714)


def test_seg_iou_loss_sigmoid(seg_pred, seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=True)
    assert iou_loss(seg_pred, seg_label).item() == _approx(0.7773743)


def test_seg_iou_loss_no_sigmoid(seg_pred, seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=False)
    assert iou_loss(seg_pred, seg_label).item() == _approx(0.5083333)
    

def test_seg_iou_loss(seg_pred, seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=False, reduction_dim=(0, 2, 3))
    assert iou_loss(seg_pred, seg_label).tolist() == _approx([0.482756, 0.458334])


def test_seg_iou_loss_with_mask(seg_pred, seg_label):
    pass
