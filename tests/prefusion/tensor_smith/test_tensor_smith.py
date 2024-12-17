import pytest
import numpy as np
import torch
import cv2
from easydict import EasyDict as edict

from prefusion.dataset.transform import Bbox3D, ParkingSlot3D, Polyline3D, OccSdfBev
from prefusion.dataset.tensor_smith import (
    get_bev_intrinsics, 
    is_in_bbox3d,
    CameraImageTensor,
    PlanarBbox3D,
    PlanarRectangularCuboid,
    PlanarParkingSlot3D,
    PlanarSquarePillar,
    PlanarCylinder3D,
    PlanarOrientedCylinder3D,
    PlanarPolyline3D,
    PlanarOccSdfBev,
)

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


def test_planar_bbox_3d_get_roll_from_xyvecs():
    a = [1, 1, 0]
    b = [-1, 1, 1]
    result = np.array(PlanarBbox3D._get_roll_from_xyvecs(a, b))
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
    delta_ij = np.float32([0.7, 0.2, 0])
    sizes = np.float32([2, 1, 0.5])
    xvec = np.float32([1, 0, 0])
    yvec = np.float32([0, 1, 0])
    zvec = np.float32([0, 0, 1])
    assert is_in_bbox3d(delta_ij, sizes, xvec, yvec, zvec) is True



def test_planar_bbox_3d_generation_and_reverse():
    pbox3d = PlanarBbox3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )

    box3d = Bbox3D(
        "bbox_3d",
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
        dictionary={'classes': ['car', 'bus']},
        tensor_smith=pbox3d
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['seg'][0].max() == 1
    pred_bboxes_3d = pbox3d.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['size'],
        box3d.elements[0]['size'],
    decimal=3)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[1]['rotation'],
        box3d.elements[1]['rotation'],
    decimal=3)



def test_planar_rectangular_cuboid_generation_and_reverse():
    prect = PlanarRectangularCuboid(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )

    box3d = Bbox3D(
        "cuboid",
        elements=[
            {
                'class': 'speedbump',
                'attr': {},
                'size': [5, 0.5, 0.1],
                'rotation': np.float32([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]),
                'translation': np.float32([
                    [8], [4], [0]
                ]),
            }
        ],
        dictionary={'classes': ['speedbump']},
        tensor_smith=prect
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['seg'][0].max() == 1
    pred_bboxes_3d = prect.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['size'],
        box3d.elements[0]['size'],
    decimal=3)
    del_R = pred_bboxes_3d[0]['rotation'].T @ box3d.elements[0]['rotation']
    np.testing.assert_almost_equal(abs(del_R[0, 0]), 1, decimal=3)



def test_planar_square_pillars_generation_and_reverse():
    psp = PlanarSquarePillar(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8]),
        use_bottom_center=True
    )
    box3d = Bbox3D(
        "bbox_3d",
        elements=[
            {
                'class': 'pillar',
                'attr': {},
                'size': [1, 1, 4],
                'rotation': np.array([
                    [ 0.8658935 , -0.48977932,  0.1017087 ],
                    [ 0.49992385,  0.85438382, -0.1417901 ],
                    [-0.01745241,  0.17362173,  0.98465776]
                ]),
                # 'rotation': np.array([
                #     [ 0.49 , -0.872, -0.   ],
                #     [ 0.872,  0.49 ,  0.   ],
                #     [-0.   , -0.   ,  1.   ]
                # ]),
                'translation': np.float32([
                    [8], [4], [0]
                ]),
                'velocity': np.float32([
                    [0], [0], [0]
                ]),
            },
        ],
        dictionary={'classes': ['pillar']},
        tensor_smith=psp
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['seg'][0].max() == 1
    pred_bboxes_3d = psp.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['size'],
        box3d.elements[0]['size'],
    decimal=3)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['rotation'],
        box3d.elements[0]['rotation'],
    decimal=3)



def test_planar_cylinder3d_generation_and_reverse():
    pc = PlanarCylinder3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8]),
        use_bottom_center=True
    )
    box3d = Bbox3D(
        "bbox_3d",
        elements=[
            {
                'class': 'cylinder_pillar',
                'attr': {},
                'size': [1, 1, 4],
                'rotation': np.array([
                    [ 0.8658935 , -0.48977932,  0.1017087 ],
                    [ 0.49992385,  0.85438382, -0.1417901 ],
                    [-0.01745241,  0.17362173,  0.98465776]
                ]),
                'translation': np.float32([
                    [8], [4], [0]
                ]),
                'velocity': np.float32([
                    [0], [0], [0]
                ]),
            },
        ],
        dictionary={'classes': ['cylinder_pillar']},
        tensor_smith=pc
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['seg'][0].max() == 1
    assert (tensor_dict['cen'][0].min() >= 0) & (tensor_dict['cen'][0].max() <= 1)
    pred_bboxes_3d = pc.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['height'],
        box3d.elements[0]['size'][2],
    decimal=3)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['zvec'],
        box3d.elements[0]['rotation'][:, 2],
    decimal=3)



def test_planar_oriented_cylinder3d_generation_and_reverse():
    poc = PlanarOrientedCylinder3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8]),
        use_bottom_center=True
    )
    box3d = Bbox3D(
        "bbox_3d",
        elements=[
            {
                'class': 'pedestrain',
                'attr': {},
                'size': [0.5, 0.5, 1.75],
                'rotation': np.array([
                    [ 0.8658935 , -0.48977932,  0.1017087 ],
                    [ 0.49992385,  0.85438382, -0.1417901 ],
                    [-0.01745241,  0.17362173,  0.98465776]
                ]),
                'translation': np.float32([
                    [8], [4], [0]
                ]),
                'velocity': np.float32([
                    [0], [0], [0]
                ]),
            },
        ],
        dictionary={'classes': ['pedestrain']},
        tensor_smith=poc
    )
    box3d.to_tensor()
    tensor_dict = box3d.tensor
    assert tensor_dict['seg'][0].max() == 1
    assert (tensor_dict['cen'][0].min() >= 0) & (tensor_dict['cen'][0].max() <= 1)
    pred_bboxes_3d = poc.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['size'],
        box3d.elements[0]['size'],
    decimal=3)
    np.testing.assert_almost_equal(
        pred_bboxes_3d[0]['rotation'],
        box3d.elements[0]['rotation'],
    decimal=3)




def test_planar_parkingslot_3d_generation_and_reverse():
    pslot = PlanarParkingSlot3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )
    slots = ParkingSlot3D(
        "parkingslot_3d",
        elements=[{
            'class': 'class.parking.paring_slot',
            'attr': {},
            'points': np.float32([
                [0, 0.9, 0.1],
                [2.7, 1, 0.1],
                [2.6, 6.4, 0.5],
                [-0.2, 6.4, 0.9]
            ])
        }],
        dictionary=dict(
            classes=['class.parking.paring_slot']
        ),
        tensor_smith=pslot
    )
    slots.to_tensor()
    tensor_dict = slots.tensor
    assert tensor_dict['seg'][0].max() == 1
    pred_slots = pslot.reverse(tensor_dict)
    np.testing.assert_almost_equal(
        pred_slots[0][:, :3],
        slots.elements[0]['points'],
    decimal=3)


@pytest.fixture()
def plyl3d() -> Polyline3D:
    return Polyline3D(
        "polyline_3d",
        elements=[
            {
                'class': 'class.road_marker.lane_line',
                'attr': {},
                'points': np.float32([
                    [-1,  2, -0.1],
                    [ 0,  1,  0.0],
                    [ 1,  0,  0.1]])
                ###################### 
                #           .
                #         /
                #       .
                #     /
                #   .
                ######################
            },
            {
                'class': 'class.road_marker.arrow_heading_triangle',
                'attr': {},
                'points': np.float32([
                    [ 0,     0,  0.0],
                    [ 1,  -0.9,  0.2],
                    [ 1,  -1.1,  0.2],
                    [ 0,    -2,  0.0]])
                ###################### 
                #       . - .
                #      /     \
                #     .       .
                ######################
            },
        ],
        dictionary=dict(
            classes=['class.road_marker.lane_line', 'class.road_marker.arrow_heading_triangle']
        ),
    )


def test_planar_polyline_3d_generation_and_reverse(plyl3d):
    plyl3d_tensor_smith = PlanarPolyline3D(
        voxel_shape=(6, 160, 80),
        voxel_range=([-0.5, 2.5], [24, -8], [8, -8])
    )
    tensor_dict = plyl3d_tensor_smith(plyl3d)
    assert tensor_dict['seg'].shape == (3, 160, 80)
    assert tensor_dict['seg'][0].max() == 1
    pred_plyl = plyl3d_tensor_smith.reverse(tensor_dict)
    np.testing.assert_almost_equal(pred_plyl[0][[0, -1], :3], plyl3d.elements[0]['points'][[0, -1], :], decimal=3)
    np.testing.assert_almost_equal(pred_plyl[1][[0, -1], :3], plyl3d.elements[1]['points'][[0, -1], :], decimal=3)


def generate_points_of_spiral_pattern(num_pts, period=8, x_scale=7.5, y_scale=7.5, x_translate=0, y_translate=0):
    # Generate a spiral pattern
    t = np.linspace(0, period * np.pi, num_pts)
    x = t * np.cos(t) / (period * np.pi) * x_scale + x_translate
    y = t * np.sin(t) / (period * np.pi) * y_scale + y_translate
    z = np.random.random(num_pts)

    # Combine coordinates into a single array
    points = np.column_stack((x, y, z))

    return points

def test_planar_polyline_3d_link_polyline():
    pts = generate_points_of_spiral_pattern(200, x_scale=12, x_translate=6)
    dic = dict(classes=['lane_line'])
    spiral = Polyline3D( "polyline_3d", elements=[ { 'class': 'lane_line', 'attr': {}, 'points': pts } ], dictionary=dic)
    plyl3d_tensor_smith = PlanarPolyline3D(voxel_shape=(6, 160, 80), voxel_range=([-0.5, 2.5], [24, -8], [8, -8]))
    tensor_dict = plyl3d_tensor_smith(spiral)
    assert tensor_dict['seg'].shape == (2, 160, 80)
    assert tensor_dict['seg'][0].max() == 1

    pred_plyl = plyl3d_tensor_smith.reverse(tensor_dict)


def test_planar_occ_sdf_bev():
    src_voxel_range = [[-1, 3], [-12.8, 38.4], [25.6, -25.6]]
    dst_voxel_range = ([-1, 3], [12, -12], [9, -9])
    occ_path = 'tests/prefusion/tensor_smith/occ_sdf_bev/occ.png'
    height_path = 'tests/prefusion/tensor_smith/occ_sdf_bev/height.tif'
    planar_occ_sdf_bev = PlanarOccSdfBev(
        voxel_shape=(6, 160, 120),
        voxel_range=dst_voxel_range,
        sdf_range=(-0.1, 5)
    )
    occ_sdf_bev = OccSdfBev(
        name='test',
        src_voxel_range=src_voxel_range,  # ego system,
        occ=cv2.imread(occ_path),
        sdf=None,
        height=cv2.imread(height_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 3000 - 10,
        mask=None,
        tensor_smith=planar_occ_sdf_bev,
    )
    occ_sdf_bev.to_tensor()
    tensor_dict = occ_sdf_bev.tensor
    assert tensor_dict['seg'].shape == (4, 160, 120)
    assert tensor_dict['reg'].shape == (2, 160, 120)
