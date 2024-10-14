import functools

import pytest
import torch
import numpy as np

from prefusion.loss.basic import SegIouLoss, seg_iou
from prefusion.loss.planar import PlanarBbox3DLoss

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


@pytest.fixture
def seg_mask():
    return torch.tensor([[
        [[1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],],
         
        [[0, 0, 0, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 1, 1],],
    ]])


@pytest.fixture
def cen_pred():
    return torch.tensor([[
        [[-79, -10, 0.0, -10, -5],
         [-80, 0.0,  10, 0.0, -5],
         [-70, 0.0,  10, 0.0, -5],
         [-60, -10, 0.0, -10, -5],]
    ]])


@pytest.fixture
def cen_label():
    return torch.tensor([[
        [[0, 0.1, 0.5, 0.1, 0],
         [0, 0.5, 1.0, 0.5, 0],
         [0, 0.5, 1.0, 0.5, 0],
         [0, 0.1, 0.5, 0.1, 0],]
    ]])


@pytest.fixture
def reg_pred():
    return torch.tensor([[
        [[0, 0.0, 0.0, 0.0, 0],
         [0, 0.0, 0.5, 0.0, 0],
         [0, 0.0, 1.2, 3.1, 0],
         [0, 0.0, 0.0, 0.0, 0],],

        [[0.9, 0.0, 0.0, 0.0, 0.1],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.9, 0.0, 0.0, 0.0, 2.0],],
    ]]).repeat((1, 10, 1, 1))


@pytest.fixture
def reg_label():
    return torch.tensor([[
        [[0, 0.0, 0.0, 0.0, 0],
         [0, 0.0, 0.8, 0.0, 0],
         [0, 0.0, 1.5, 3.4, 0],
         [0, 0.0, 0.0, 0.0, 0],],

        [[0.9, 0.0, 0.0, 0.0, 0.7],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [1.1, 0.0, 0.0, 0.0, 2.0],],
    ]]).repeat((1, 10, 1, 1))


@pytest.fixture
def reg_channel_weights():
    return {
        "center_xy_weight": 1.0,
        "center_z_weight": 0.0,
        "size_weight": 0.0,
        "unit_xvec_weight": 0.0,
        "abs_xvec_weight": 0.0,
        "xvec_product_weight": 2.0,
        "abs_roll_angle_weight": 0.0,
        "roll_angle_product_weight": 0.0,
        "velo_weight": 0.0,
    }


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


def test_seg_iou_loss_sigmoid_log(seg_pred, seg_label):
    iou_loss = SegIouLoss(method="log", pred_logits=True, reduction_dim=(0, 2, 3))
    assert iou_loss(seg_pred, seg_label).tolist() == _approx([1.468862, 1.298853])


def test_seg_iou_loss_with_mask(seg_pred, seg_label, seg_mask):
    iou_loss = SegIouLoss(method="linear", pred_logits=False, reduction_dim=(0, 2, 3))
    assert iou_loss(seg_pred, seg_label, mask=seg_mask).tolist() == _approx([0.425, 0.5106383])


def test_planar_bbox3d_seg_loss(seg_pred, seg_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d", class_weights=[0.3, 0.7])
    seg_loss = planar_bbox3d_loss._seg_loss(seg_pred, seg_label)
    assert set(seg_loss.keys()) == {
        "plnrbox3d_seg_iou_0_loss", "plnrbox3d_seg_iou_1_loss", "plnrbox3d_seg_iou_loss",
        "plnrbox3d_seg_dual_focal_0_loss", "plnrbox3d_seg_dual_focal_1_loss", "plnrbox3d_seg_dual_focal_loss",
        "plnrbox3d_seg_loss"
    }
    assert seg_loss["plnrbox3d_seg_iou_0_loss"] == _approx(0.4406586)
    assert seg_loss["plnrbox3d_seg_iou_1_loss"] == _approx(0.909197)
    assert seg_loss["plnrbox3d_seg_iou_loss"] == _approx(1.3498556)
    assert seg_loss["plnrbox3d_seg_dual_focal_0_loss"] == _approx(0.359088)
    assert seg_loss["plnrbox3d_seg_dual_focal_1_loss"] == _approx(0.83189)
    assert seg_loss["plnrbox3d_seg_dual_focal_loss"] == _approx(1.190978)
    assert seg_loss["plnrbox3d_seg_loss"] == _approx(2.5408336)


def test_planar_bbox3d_cen_loss(cen_pred, cen_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    cen_mask = torch.zeros((1, 1, 4, 5))
    cen_mask[:, :, :, 1:4] = 1
    cen_loss = planar_bbox3d_loss._cen_loss(cen_pred, cen_label, cen_mask)
    assert set(cen_loss.keys()) == {"plnrbox3d_cen_dual_focal_loss", "plnrbox3d_cen_fg_dual_focal_loss", "plnrbox3d_cen_loss"}
    assert cen_loss["plnrbox3d_cen_dual_focal_loss"] == _approx(0.430634)
    assert cen_loss["plnrbox3d_cen_fg_dual_focal_loss"] == _approx(0.7132555)
    assert cen_loss["plnrbox3d_cen_loss"] == _approx(1.1438895)

def test_planar_bbox3d_reg_loss(reg_pred, reg_label, reg_channel_weights):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    reg_mask = torch.ones((1, 1, 4, 5))
    reg_loss = planar_bbox3d_loss._reg_loss(reg_pred, reg_label, reg_mask, **reg_channel_weights)
    assert set(reg_loss.keys()) == {
        "plnrbox3d_reg_center_xy_loss",
        "plnrbox3d_reg_center_z_loss",
        "plnrbox3d_reg_size_loss",
        "plnrbox3d_reg_unit_xvec_loss",
        "plnrbox3d_reg_abs_xvec_loss",
        "plnrbox3d_reg_xvec_product_loss",
        "plnrbox3d_reg_abs_roll_angle_loss",
        "plnrbox3d_reg_roll_angle_product_loss",
        "plnrbox3d_reg_velo_loss",
        "plnrbox3d_reg_loss",
    }
    assert reg_loss["plnrbox3d_reg_center_xy_loss"] == _approx(0.0425)
    assert reg_loss["plnrbox3d_reg_xvec_product_loss"] == _approx(0.085)
    assert reg_loss["plnrbox3d_reg_loss"] == _approx(0.1275)


def test_planar_bbox3d_reg_loss_with_mask(reg_pred, reg_label, reg_channel_weights):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    reg_mask = torch.zeros((1, 1, 4, 5))
    reg_mask[:, :, 2:, :] = 1
    reg_loss = planar_bbox3d_loss._reg_loss(reg_pred, reg_label, reg_mask, **reg_channel_weights)
    assert reg_loss["plnrbox3d_reg_center_xy_loss"] == _approx(0.04)
    assert reg_loss["plnrbox3d_reg_xvec_product_loss"] == _approx(0.08)
    assert reg_loss["plnrbox3d_reg_loss"] == _approx(0.12)


def test_planar_bbox3d_loss(seg_pred, cen_pred, reg_pred, seg_label, cen_label, reg_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d", class_weights=[0.3, 0.7])
    pred = {"seg": seg_pred, "cen": cen_pred, "reg": reg_pred}
    label = {"seg": seg_label, "cen": cen_label, "reg": reg_label}
    loss = planar_bbox3d_loss(pred, label)
    assert set(loss.keys()) == {
        "plnrbox3d_seg_iou_0_loss", "plnrbox3d_seg_iou_1_loss", "plnrbox3d_seg_iou_loss",
        "plnrbox3d_seg_dual_focal_0_loss", "plnrbox3d_seg_dual_focal_1_loss", "plnrbox3d_seg_dual_focal_loss",
        "plnrbox3d_seg_loss", "plnrbox3d_cen_dual_focal_loss", "plnrbox3d_cen_fg_dual_focal_loss", "plnrbox3d_cen_loss",
        "plnrbox3d_reg_center_xy_loss", "plnrbox3d_reg_center_z_loss", "plnrbox3d_reg_size_loss", "plnrbox3d_reg_unit_xvec_loss",
        "plnrbox3d_reg_abs_xvec_loss", "plnrbox3d_reg_xvec_product_loss", "plnrbox3d_reg_abs_roll_angle_loss", 
        "plnrbox3d_reg_roll_angle_product_loss", "plnrbox3d_reg_velo_loss", "plnrbox3d_reg_loss",
        "plnrbox3d_loss",
    }
    assert loss["plnrbox3d_seg_iou_0_loss"] == _approx(0.4406586)
    assert loss["plnrbox3d_seg_iou_1_loss"] == _approx(0.909197)
    assert loss["plnrbox3d_seg_iou_loss"] == _approx(1.3498556)
    assert loss["plnrbox3d_seg_dual_focal_0_loss"] == _approx(0.359088)
    assert loss["plnrbox3d_seg_dual_focal_1_loss"] == _approx(0.83189)
    assert loss["plnrbox3d_seg_dual_focal_loss"] == _approx(1.190978)
    assert loss["plnrbox3d_seg_loss"] == _approx(2.5408336)
    
    assert loss["plnrbox3d_cen_dual_focal_loss"] == _approx(0.430634)
    assert loss["plnrbox3d_cen_fg_dual_focal_loss"] == _approx(0.2311096)
    assert loss["plnrbox3d_cen_loss"] == _approx(0.6617445)

    assert loss["plnrbox3d_reg_center_xy_loss"] == _approx(0.15)  # 0:2
    assert loss["plnrbox3d_reg_center_z_loss"] == _approx(0.3)  # 2:3
    assert loss["plnrbox3d_reg_size_loss"] == _approx(0.1)  # 3:6
    assert loss["plnrbox3d_reg_unit_xvec_loss"] == _approx(0.2)  # 6:9
    assert loss["plnrbox3d_reg_abs_xvec_loss"] == _approx(0.1)  # 9:12
    assert loss["plnrbox3d_reg_xvec_product_loss"] == _approx(0.15)  # 12:14
    assert loss["plnrbox3d_reg_abs_roll_angle_loss"] == _approx(0.15)  # 14:16
    assert loss["plnrbox3d_reg_roll_angle_product_loss"] == _approx(0.3)  # 16:17
    assert loss["plnrbox3d_reg_velo_loss"] == _approx(0.1)  # 17:20
    assert loss["plnrbox3d_reg_loss"] == _approx(1.55)
    
    assert loss["plnrbox3d_loss"] == _approx(4.7525781)