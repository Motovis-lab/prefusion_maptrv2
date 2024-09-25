import pytest
import numpy as np
import torch
from easydict import EasyDict as edict

from prefusion.dataset.tensor_smith import get_bev_intrinsics, CameraImageTensor, PlanarBbox3D
from prefusion.dataset.transform import Bbox3D, ToTensor

def test_get_bev_intrinsics():
    voxel_shape=(6, 200, 160)
    voxel_range=([-0.5, 2.5], [10, -10], [8, -8])
    bev_intrinsics = get_bev_intrinsics(voxel_shape, voxel_range)
    assert bev_intrinsics == (99.5, 79.5, -10, -10)


def test_camera_image_tensor():
    camera_image = edict(
        img=np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(np.uint8),
        ego_mask=np.ones((2, 4, 3), dtype=np.uint8)
    )
    tensor_smith = CameraImageTensor(means=[1, 2, 3], stds=[0.1, 0.2, 0.3])
    tensor_dict = tensor_smith(camera_image)
    np.testing.assert_almost_equal(tensor_dict["img"].numpy(), np.array([
        [[-10.,  20.,  50.,  80.],
         [110., 140., 170., 200.]],

        [[ -5.,  10.,  25.,  40.],
         [ 55.,  70.,  85., 100.]],

        [[-3.33333333,  6.66666667, 16.66666667, 26.66666667],
         [36.66666667, 46.66666667, 56.66666667, 66.66666667]]
    ]), decimal=6)



def test_planar_bbox_3d_get_roll_from_vecs():
    a = [1, 1, 0]
    b = [-1, 1, 1]
    result = np.array(PlanarBbox3D._get_roll_from_vecs(a, b))
    answer = np.array((0.8164965669539823, 0.5773502593086969))
    np.testing.assert_almost_equal(result, answer, decimal=6)


def test_planar_bbox_3d_get_yzvec_from_xvec_and_roll():
    xvecs = np.float32([
        [1, 1], 
        [1, 1],
        [0, 0]
    ])
    roll_vecs = np.array([
        [0.8164965669539823, 1],  # cos
        [0.5773502593086969, 0]   # sin
    ])
    yvecs, zvecs = PlanarBbox3D._get_yzvec_from_xvec_and_roll(xvecs, roll_vecs)
    yvec_ans = np.float32([-1, 1, 1])
    yvec_ans /= np.linalg.norm(yvec_ans)
    np.testing.assert_almost_equal(yvecs[:, 0], yvec_ans, decimal=6)


def test_planar_bbox_3d_get_yzvec_from_xvec_and_roll_single():
    xvec = np.float32([1, 1, 0])
    roll_vec = np.array([0.8164965669539823, 0.5773502593086969])
    yvec, zvec = PlanarBbox3D._get_yzvec_from_xvec_and_roll(xvec, roll_vec)
    yvec_ans = np.float32([-1, 1, 1])
    yvec_ans /= np.linalg.norm(yvec_ans)
    np.testing.assert_almost_equal(yvec, yvec_ans, decimal=6)



def test_planar_bbox_3d_is_in_bbox3d():
    pbox3d = PlanarBbox3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )
    delta_ij = np.float32([0.7, 0.2, 0])
    sizes = np.float32([2, 1, 0.5])
    xvec = np.float32([1, 0, 0]) 
    roll = np.float32([0.8164965669539823, 0.5773502593086969])
    assert pbox3d._is_in_bbox3d(delta_ij, sizes, xvec, roll) is True



def test_planar_bbox_3d_generation_and_reverse():
    pbox3d = PlanarBbox3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )

    box3d = Bbox3D(
        elements=[
            {
                'class': 'bus',
                'attr': {},
                'size': [10, 2.5, 3.0],
                'rotation': np.float32([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'translation': np.float32([
                    [8], [4], [0]
                ]),
                'velocity': np.float32([
                    [2], [0], [0]
                ]),
            },
            {
                'class': 'car',
                'attr': {},
                'size': [5, 2, 1.6],
                'rotation': np.float32([
                    [ 0.8, 0.6, 0],
                    [-0.6, 0.8, 0],
                    [ 0  , 0  , 1]
                ]),
                'translation': np.float32([
                    [3], [-5], [0]
                ]),
                'velocity': np.float32([
                    [3], [-1], [0]
                ]),
            },
        ],
        dictionary={
            'branch_0': {
                'classes': ['car', 'bus']
            },
            'branch_1': {
                'classes': ['car'],
            }
        },
        tensor_smith=pbox3d
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['branch_0']['seg'][0].max() == 1
    pred_bboxes_3d = pbox3d.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d['branch_0'][0]['size'],
        box3d.elements[0]['size'],
    decimal=3)
    np.testing.assert_almost_equal(
        pred_bboxes_3d['branch_1'][0]['rotation'],
        box3d.elements[1]['rotation'],
    decimal=3)