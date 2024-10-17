import functools

import pytest
import torch
import numpy as np

from prefusion.loss.basic import SegIouLoss, seg_iou
from prefusion.loss.planar import PlanarBbox3DLoss, PlanarPolyline3DLoss

_approx = functools.partial(pytest.approx, rel=1e-5)

@pytest.fixture
def bx_seg_pred():
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
def bx_seg_label():
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
def bx_seg_mask():
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
def bx_cen_pred():
    return torch.tensor([[
        [[-79, -10, 0.0, -10, -5],
         [-80, 0.0,  10, 0.0, -5],
         [-70, 0.0,  10, 0.0, -5],
         [-60, -10, 0.0, -10, -5],]
    ]])


@pytest.fixture
def bx_cen_label():
    return torch.tensor([[
        [[0, 0.1, 0.5, 0.1, 0],
         [0, 0.5, 1.0, 0.5, 0],
         [0, 0.5, 1.0, 0.5, 0],
         [0, 0.1, 0.5, 0.1, 0],]
    ]])


@pytest.fixture
def bx_reg_pred():
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
def bx_reg_label():
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
def bx_reg_channel_weights():
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


def test_seg_iou(bx_seg_pred, bx_seg_label):
    assert seg_iou(bx_seg_pred[0, 0], bx_seg_label[0, 0]).item() == _approx(0.51724)
    assert seg_iou(bx_seg_pred[0, 1], bx_seg_label[0, 1]).item() == _approx(0.541666)
    assert seg_iou(bx_seg_pred, bx_seg_label, dim=(0, 2, 3)).tolist() == _approx([0.51724, 0.541666])
    assert seg_iou(bx_seg_pred[0, 1], torch.zeros_like(bx_seg_pred[0, 1])).item() == _approx(0.75714)


def test_seg_iou_loss_sigmoid(bx_seg_pred, bx_seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=True)
    assert iou_loss(bx_seg_pred, bx_seg_label).item() == _approx(0.7773743)


def test_seg_iou_loss_no_sigmoid(bx_seg_pred, bx_seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=False)
    assert iou_loss(bx_seg_pred, bx_seg_label).item() == _approx(0.5083333)
    

def test_seg_iou_loss(bx_seg_pred, bx_seg_label):
    iou_loss = SegIouLoss(method="linear", pred_logits=False, reduction_dim=(0, 2, 3))
    assert iou_loss(bx_seg_pred, bx_seg_label).tolist() == _approx([0.482756, 0.458334])


def test_seg_iou_loss_sigmoid_log(bx_seg_pred, bx_seg_label):
    iou_loss = SegIouLoss(method="log", pred_logits=True, reduction_dim=(0, 2, 3))
    assert iou_loss(bx_seg_pred, bx_seg_label).tolist() == _approx([1.468862, 1.298853])


def test_seg_iou_loss_with_mask(bx_seg_pred, bx_seg_label, bx_seg_mask):
    iou_loss = SegIouLoss(method="linear", pred_logits=False, reduction_dim=(0, 2, 3))
    assert iou_loss(bx_seg_pred, bx_seg_label, mask=bx_seg_mask).tolist() == _approx([0.425, 0.5106383])


def test_planar_bbox3d_seg_loss(bx_seg_pred, bx_seg_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d", class_weights=[0.3, 0.7])
    seg_loss = planar_bbox3d_loss._seg_loss(bx_seg_pred, bx_seg_label)
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


def test_planar_bbox3d_cen_loss(bx_cen_pred, bx_cen_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    cen_mask = torch.zeros((1, 1, 4, 5))
    cen_mask[:, :, :, 1:4] = 1
    cen_loss = planar_bbox3d_loss._cen_loss(bx_cen_pred, bx_cen_label, cen_mask)
    assert set(cen_loss.keys()) == {"plnrbox3d_cen_dual_focal_loss", "plnrbox3d_cen_fg_dual_focal_loss", "plnrbox3d_cen_loss"}
    assert cen_loss["plnrbox3d_cen_dual_focal_loss"] == _approx(0.430634)
    assert cen_loss["plnrbox3d_cen_fg_dual_focal_loss"] == _approx(0.7132555)
    assert cen_loss["plnrbox3d_cen_loss"] == _approx(1.1438895)

def test_planar_bbox3d_reg_loss(bx_reg_pred, bx_reg_label, bx_reg_channel_weights):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    reg_mask = torch.ones((1, 1, 4, 5))
    reg_loss = planar_bbox3d_loss._reg_loss(bx_reg_pred, bx_reg_label, reg_mask, **bx_reg_channel_weights)
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


def test_planar_bbox3d_reg_loss_with_mask(bx_reg_pred, bx_reg_label, bx_reg_channel_weights):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d")
    reg_mask = torch.zeros((1, 1, 4, 5))
    reg_mask[:, :, 2:, :] = 1
    reg_loss = planar_bbox3d_loss._reg_loss(bx_reg_pred, bx_reg_label, reg_mask, **bx_reg_channel_weights)
    assert reg_loss["plnrbox3d_reg_center_xy_loss"] == _approx(0.04)
    assert reg_loss["plnrbox3d_reg_xvec_product_loss"] == _approx(0.08)
    assert reg_loss["plnrbox3d_reg_loss"] == _approx(0.12)


def test_planar_bbox3d_loss(bx_seg_pred, bx_cen_pred, bx_reg_pred, bx_seg_label, bx_cen_label, bx_reg_label):
    planar_bbox3d_loss = PlanarBbox3DLoss(loss_name_prefix="plnrbox3d", class_weights=[0.3, 0.7])
    pred = {"seg": bx_seg_pred, "cen": bx_cen_pred, "reg": bx_reg_pred}
    label = {"seg": bx_seg_label, "cen": bx_cen_label, "reg": bx_reg_label}
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


@pytest.fixture
def plyl_seg_pred():
    return torch.tensor([[
        [[0.0, 0.0, 0.9, 0.0, 0.3],
         [0.0, 0.0, 0.3, 1.0, 0.0],
         [0.2, 1.0, 1.0, 1.0, 0.0],
         [0.0, 0.9, 0.0, 0.2, 0.0],],

        [[0.0, 0.0, 0.0, 0.0, 0.3],
         [0.0, 0.0, 0.2, 1.0, 0.0],
         [0.0, 0.2, 1.0, 0.2, 0.0],
         [0.0, 0.9, 0.2, 0.0, 0.0],],
         
        [[0.0, 0.0, 0.8, 0.0, 0.0],
         [0.0, 0.0, 0.2, 0.0, 0.0],
         [0.2, 1.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.2, 0.0],],
    ]])


@pytest.fixture
def plyl_seg_label():
    return torch.tensor([[
        [[0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 1.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0],],

        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0],],
         
        [[0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],],
    ]])


@pytest.fixture
def plyl_reg_pred():
    return torch.tensor([[
        [[0, 0.0, 0.3, 0.0, 0],
         [0, 0.0, 0.0, 0.1, 0],
         [0, 0.2, 0.1, 0.1, 0],
         [0, 0.1, 0.0, 0.0, 0],],

        [[0, 0.0, -0.1, 0.0, 0],
         [0, 0.0,  0.0, 0.1, 0],
         [0, 0.3,  0.1, 0.2, 0],
         [0, 0.1,  0.0, 0.0, 0],],
    ]]).repeat((1, 4, 1, 1))[:, :7, ...]


@pytest.fixture
def plyl_reg_label():
    return torch.tensor([[
        [[0, 0.0, 0.2, 0.0, 0],
         [0, 0.0, 0.0, 0.1, 0],
         [0, 0.2, 0.1, 0.2, 0],
         [0, 0.1, 0.0, 0.0, 0],],

        [[0, 0.0, -0.2, 0.0, 0],
         [0, 0.0,  0.0, 0.1, 0],
         [0, 0.2,  0.1, 0.2, 0],
         [0, 0.1,  0.0, 0.0, 0],],
    ]]).repeat((1, 4, 1, 1))[:, :7, ...]


@pytest.fixture
def plyl_reg_channel_weights():
    return {
        "dist_weight": 1.0,
        "vert_vec_weight": 2.0,
        "abs_dir_weight": 0.0,
        "dir_product_weight": 0.0,
        "height_weight": 0.0,
    }


def test_planar_polyline3d_seg_loss_with_mask(plyl_seg_pred, plyl_seg_label):
    planar_polyline3d_loss = PlanarPolyline3DLoss(loss_name_prefix="plnrplyln3d")
    seg_loss = planar_polyline3d_loss._seg_loss(plyl_seg_pred, plyl_seg_label)
    assert set(seg_loss.keys()) == {
        "plnrplyln3d_seg_iou_0_loss", "plnrplyln3d_seg_iou_1_loss", "plnrplyln3d_seg_iou_2_loss",
        "plnrplyln3d_seg_iou_loss", "plnrplyln3d_seg_dual_focal_0_loss", "plnrplyln3d_seg_dual_focal_1_loss", 
        "plnrplyln3d_seg_dual_focal_2_loss", "plnrplyln3d_seg_dual_focal_loss", "plnrplyln3d_seg_loss"
    }
    assert seg_loss["plnrplyln3d_seg_iou_0_loss"] == _approx(0.980281949)
    assert seg_loss["plnrplyln3d_seg_loss"] == _approx(2.3641901)


def test_planar_polyline3d_reg_loss_with_mask(plyl_reg_pred, plyl_reg_label, plyl_reg_channel_weights):
    planar_polyline3d_loss = PlanarPolyline3DLoss(loss_name_prefix="plnrplyln3d")
    reg_mask = torch.zeros((1, 1, 4, 5))
    reg_mask[:, :, :, 1:4] = 1
    reg_loss = planar_polyline3d_loss._reg_loss(plyl_reg_pred, plyl_reg_label, reg_mask, **plyl_reg_channel_weights)
    assert set(reg_loss.keys()) == {
        "plnrplyln3d_reg_dist_loss",
        "plnrplyln3d_reg_vert_vec_loss",
        "plnrplyln3d_reg_abs_dir_loss",
        "plnrplyln3d_reg_dir_product_loss",
        "plnrplyln3d_reg_height_loss",
        "plnrplyln3d_reg_loss",
    }
    assert reg_loss["plnrplyln3d_reg_dist_loss"] == _approx(0.0166666)
    assert reg_loss["plnrplyln3d_reg_vert_vec_loss"] == _approx(0.0333333)
    assert reg_loss["plnrplyln3d_reg_loss"] == _approx(0.05)
