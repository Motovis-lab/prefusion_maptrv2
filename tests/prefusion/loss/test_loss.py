import functools

import pytest
import torch

from prefusion.loss.basic import SegIouLoss, seg_iou
from prefusion.loss.planar import PlanarLoss


_approx = functools.partial(pytest.approx, rel=1e-5)


@pytest.fixture
def planar_bbox3d_loss():
    return PlanarLoss(loss_name_prefix="plnrbox3d", weight_scheme={
        "seg": {
            "loss_weight": 1.0,
            "class_weights": {
                "0": {"weight": 0.3},
                "1": {"weight": 0.7},
            },
            "iou_loss_weight": 1.0,
            "dual_focal_loss_weight": 1.0,
        },
        "cen": {
            "loss_weight": 1.0,
            "fg_weight": 1.0,
            "bg_weight": 1.0,
        },
        "reg": {
            "loss_weight": 1.0,
            "partition_weights": {
                "center_xy": {"weight": 1.0, "slice": (0, 2)},
                "center_z": {"weight": 0.0, "slice": 2},
                "size": {"weight": 0.0, "slice": (3, 6)},
                "unit_xvec": {"weight": 0.0, "slice": (6, 9)},
                "abs_xvec": {"weight": 0.0, "slice": (9, 12)},
                "xvec_product": {"weight": 2.0, "slice": (12, 14)},
                "abs_roll_angle": {"weight": 0.0, "slice": (14, 16)},
                "roll_angle_product": {"weight": 0.0, "slice": 16},
                "velo": {"weight": 0.0, "slice": (17, 20)},
            }
        }
    })


@pytest.fixture
def planar_polyline3d_loss():
    return PlanarLoss(loss_name_prefix="plnrplyl3d", weight_scheme={
        "seg": {
            "loss_weight": 1.0,
            "class_weights": {
                "0": {"weight": 1.0},
                "1": {"weight": 1.0},
            },
            "iou_loss_weight": 1.0,
            "dual_focal_loss_weight": 1.0,
        },
        "cen": {
            "loss_weight": 1.0,
            "fg_weight": 1.0,
            "bg_weight": 1.0,
        },
        "reg": {
            "loss_weight": 1.0,
            "partition_weights": {
                "dist": {"weight": 1.0, "slice": (0, 1)},
                "vert_vec": {"weight": 2.0, "slice": (1, 3)},
                "abs_dir": {"weight": 0.0, "slice": (3, 5)},
                "dir_product": {"weight": 0.0, "slice": (5, 6)},
                "height": {"weight": 0.0, "slice": (6, 7)},
            }
        }
    })

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


def test_planar_bbox3d_seg_loss(bx_seg_pred, bx_seg_label, planar_bbox3d_loss):
    _class_weights = planar_bbox3d_loss.weight_scheme.seg.class_weights
    seg_loss = planar_bbox3d_loss._seg_loss(bx_seg_pred, bx_seg_label, class_weights=_class_weights)
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


def test_planar_bbox3d_cen_loss(bx_cen_pred, bx_cen_label, planar_bbox3d_loss):
    cen_mask = torch.zeros((1, 1, 4, 5))
    cen_mask[:, :, :, 1:4] = 1
    cen_loss = planar_bbox3d_loss._cen_loss(bx_cen_pred, bx_cen_label, fg_mask=cen_mask)
    assert set(cen_loss.keys()) == {"plnrbox3d_cen_dual_focal_loss", "plnrbox3d_cen_fg_dual_focal_loss", "plnrbox3d_cen_loss"}
    assert cen_loss["plnrbox3d_cen_dual_focal_loss"] == _approx(0.430634)
    assert cen_loss["plnrbox3d_cen_fg_dual_focal_loss"] == _approx(0.7132555)
    assert cen_loss["plnrbox3d_cen_loss"] == _approx(1.1438895)

def test_planar_bbox3d_reg_loss(bx_reg_pred, bx_reg_label, planar_bbox3d_loss):
    _partition_weights = planar_bbox3d_loss.weight_scheme.reg.partition_weights
    reg_mask = torch.ones((1, 1, 4, 5))
    reg_loss = planar_bbox3d_loss._reg_loss(bx_reg_pred, bx_reg_label, fg_mask=reg_mask, partition_weights=_partition_weights)
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


def test_planar_bbox3d_reg_loss_with_mask(bx_reg_pred, bx_reg_label, planar_bbox3d_loss):
    _partition_weights = planar_bbox3d_loss.weight_scheme.reg.partition_weights
    reg_mask = torch.zeros((1, 1, 4, 5))
    reg_mask[:, :, 2:, :] = 1
    reg_loss = planar_bbox3d_loss._reg_loss(bx_reg_pred, bx_reg_label, fg_mask=reg_mask, partition_weights=_partition_weights)
    assert reg_loss["plnrbox3d_reg_center_xy_loss"] == _approx(0.04)
    assert reg_loss["plnrbox3d_reg_xvec_product_loss"] == _approx(0.08)
    assert reg_loss["plnrbox3d_reg_loss"] == _approx(0.12)


def test_planar_bbox3d_loss(bx_seg_pred, bx_cen_pred, bx_reg_pred, bx_seg_label, bx_cen_label, bx_reg_label, planar_bbox3d_loss):
    pred = {"seg": bx_seg_pred, "cen": bx_cen_pred, "reg": bx_reg_pred}
    label = {"seg": bx_seg_label, "cen": bx_cen_label, "reg": bx_reg_label}
    for pname in planar_bbox3d_loss.weight_scheme.reg.partition_weights:
        planar_bbox3d_loss.weight_scheme.reg.partition_weights[pname]["weight"] = 1.0
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


def test_planar_polyline3d_seg_loss_with_mask(plyl_seg_pred, plyl_seg_label, planar_polyline3d_loss):
    seg_loss = planar_polyline3d_loss._seg_loss(plyl_seg_pred, plyl_seg_label)
    assert set(seg_loss.keys()) == {
        "plnrplyl3d_seg_iou_0_loss", "plnrplyl3d_seg_iou_1_loss", "plnrplyl3d_seg_iou_2_loss",
        "plnrplyl3d_seg_iou_loss", "plnrplyl3d_seg_dual_focal_0_loss", "plnrplyl3d_seg_dual_focal_1_loss", 
        "plnrplyl3d_seg_dual_focal_2_loss", "plnrplyl3d_seg_dual_focal_loss", "plnrplyl3d_seg_loss"
    }
    assert seg_loss["plnrplyl3d_seg_iou_0_loss"] == _approx(0.980281949)
    assert seg_loss["plnrplyl3d_seg_loss"] == _approx(2.3641901)


def test_planar_polyline3d_reg_loss_with_mask(plyl_reg_pred, plyl_reg_label, planar_polyline3d_loss):
    reg_mask = torch.zeros((1, 1, 4, 5))
    reg_mask[:, :, :, 1:4] = 1
    _partition_weights = planar_polyline3d_loss.weight_scheme.reg.partition_weights
    reg_loss = planar_polyline3d_loss._reg_loss(plyl_reg_pred, plyl_reg_label, fg_mask=reg_mask, partition_weights=_partition_weights)
    assert set(reg_loss.keys()) == {
        "plnrplyl3d_reg_dist_loss",
        "plnrplyl3d_reg_vert_vec_loss",
        "plnrplyl3d_reg_abs_dir_loss",
        "plnrplyl3d_reg_dir_product_loss",
        "plnrplyl3d_reg_height_loss",
        "plnrplyl3d_reg_loss",
    }
    assert reg_loss["plnrplyl3d_reg_dist_loss"] == _approx(0.0166666)
    assert reg_loss["plnrplyl3d_reg_vert_vec_loss"] == _approx(0.0333333)
    assert reg_loss["plnrplyl3d_reg_loss"] == _approx(0.05)


def test_planar_polyline3d_reg_loss_wrong_slices(plyl_reg_pred, plyl_reg_label, planar_polyline3d_loss):
    _partition_weights = planar_polyline3d_loss.weight_scheme.reg.partition_weights
    for i, pname in enumerate(planar_polyline3d_loss.weight_scheme.reg.partition_weights):
        planar_polyline3d_loss.weight_scheme.reg.partition_weights[pname].slice = slice(i, i + 1)
    with pytest.raises(AssertionError):
        _ = planar_polyline3d_loss._reg_loss(plyl_reg_pred, plyl_reg_label, partition_weights=_partition_weights)


def test_enumerate_slices_1():
    slices = [slice(0, 1), slice(1, 3), slice(3, 5)]
    assert PlanarLoss.enumerate_slices(slices) == [0, 1, 2, 3, 4]


def test_enumerate_slices_2():
    slices = [slice(3, 6), slice(1, 3), slice(0, 1)]
    assert PlanarLoss.enumerate_slices(slices) == [0, 1, 2, 3, 4, 5]


def test_enumerate_slices_3():
    slices = [slice(3, 6), slice(1, 3), slice(2, 5)]
    assert PlanarLoss.enumerate_slices(slices) == [1, 2, 2, 3, 3, 4, 4, 5]
