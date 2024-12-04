from pathlib import Path

import numpy as np
import pytest

from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.group_sampler import IndexGroupSampler, generate_groups


@pytest.fixture
def scene_frame_inds():
    return {
        "20231101_160337": [
            "20231101_160337/1698825817664", # 1, 1
            "20231101_160337/1698825817764",
            "20231101_160337/1698825817864", # 2, 1   and 2, 0
            "20231101_160337/1698825817964",
            "20231101_160337/1698825818064",
            "20231101_160337/1698825818164",
            "20231101_160337/1698825818264", # 0, 0
            "20231101_160337/1698825818364",
            "20231101_160337/1698825818464",
            "20231101_160337/1698825818564",
            "20231101_160337/1698825818664", # 1, 0
            "20231101_160337/1698825818764",
            "20231101_160337/1698825818864",
            "20231101_160337/1698825818964", # 0, 1
            "20231101_160337/1698825819064",
            "20231101_160337/1698825819164",
            "20231101_160337/1698825819264",
        ],
        "20231101_160337_subset": [
            "20231101_160337_subset/1698825818164",
            "20231101_160337_subset/1698825818264",
            "20231101_160337_subset/1698825818364",
            "20231101_160337_subset/1698825818464",
            "20231101_160337_subset/1698825818564",
            "20231101_160337_subset/1698825818664",
            "20231101_160337_subset/1698825818764",
            "20231101_160337_subset/1698825818864",
        ],
        "20230823_110018": [
            "20230823_110018/1692759640764",
            "20230823_110018/1692759640864",
            "20230823_110018/1692759640964",
            "20230823_110018/1692759641064",
            "20230823_110018/1692759641164",
            "20230823_110018/1692759641264",
            "20230823_110018/1692759641364",
            "20230823_110018/1692759641464",
            "20230823_110018/1692759641564",
            "20230823_110018/1692759641664",
            "20230823_110018/1692759641764",
        ]
    }


def test_generate_groups_static_method_1a():
    groups = generate_groups(16, 5, 1, start_ind=0)
    answer = np.array(
        [[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_1b():
    groups = generate_groups(16, 4, 4, start_ind=0)
    answer = np.array(
        [[0, 4,  8, 12],
        [1, 5,  9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_2():
    groups = generate_groups(16, 5, 1, start_ind=3)
    answer = np.array(
        [[ 0,  1,  2,  3,  4],
        [ 3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12],
        [11, 12, 13, 14, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_3():
    groups = generate_groups(16, 5, 2, start_ind=0)
    answer = np.array(
        [[ 0,  2,  4,  6,  8],
        [ 1,  3,  5,  7,  9],
        [ 6,  8, 10, 12, 14],
        [ 7,  9, 11, 13, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_4a():
    groups = generate_groups(16, 5, 2, start_ind=4)
    answer = np.array(
        [[ 0,  2,  4,  6,  8],
        [ 1,  3,  5,  7,  9],
        [ 4,  6,  8, 10, 12],
        [ 5,  7,  9, 11, 13],
        [ 6,  8, 10, 12, 14],
        [ 7,  9, 11, 13, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_4b():
    groups = generate_groups(16, 5, 2, start_ind=9)
    answer = np.array(
        [[0, 2,  4,  6,  8],
        [1, 3,  5,  7,  9],
        [6, 8, 10, 12, 14],
        [7, 9, 11, 13, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_5():
    groups = generate_groups(16, 10, 2, start_ind=0, pad_mode='both')
    answer = np.array(
        [[ 0,  0,  2,  4,  6,  8, 10, 12, 14, 15],
         [ 0,  1,  3,  5,  7,  9, 11, 13, 15, 15]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_generate_groups_static_method_6():
    groups = generate_groups(17, 20, 2, start_ind=0, pad_mode='prev')
    answer = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7,  9, 11, 13, 15],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16]]
    )
    np.testing.assert_almost_equal(groups, answer)


def test_group_sampler_sample_train_groups_1scene(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = IndexGroupSampler("train", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    train_groups = gbs.sample_train_groups(_scene_frame_inds)
    assert train_groups == [ 
        ['20231101_160337/1698825818864', '20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164'],
        ['20231101_160337/1698825818064', '20231101_160337/1698825818164', '20231101_160337/1698825818264', '20231101_160337/1698825818364'],
        ['20231101_160337/1698825818464', '20231101_160337/1698825818564', '20231101_160337/1698825818664', '20231101_160337/1698825818764'],
        ['20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
        ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817864', '20231101_160337/1698825817964'],
    ]

def test_group_sampler_sample_train_groups_more_scenes(scene_frame_inds):
    gbs = IndexGroupSampler("train", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    train_groups = gbs.sample_train_groups(scene_frame_inds)
    assert train_groups == [
        ['20230823_110018/1692759640764', '20230823_110018/1692759640864', '20230823_110018/1692759640964', '20230823_110018/1692759641064'],
        ['20231101_160337/1698825818864', '20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164'],
        ['20231101_160337/1698825818464', '20231101_160337/1698825818564', '20231101_160337/1698825818664', '20231101_160337/1698825818764'],
        ['20230823_110018/1692759641164', '20230823_110018/1692759641264', '20230823_110018/1692759641364', '20230823_110018/1692759641464'],
        ['20231101_160337_subset/1698825818164', '20231101_160337_subset/1698825818264', '20231101_160337_subset/1698825818364', '20231101_160337_subset/1698825818464'],
        ['20231101_160337_subset/1698825818564', '20231101_160337_subset/1698825818664', '20231101_160337_subset/1698825818764', '20231101_160337_subset/1698825818864'],
        ['20230823_110018/1692759641464', '20230823_110018/1692759641564', '20230823_110018/1692759641664', '20230823_110018/1692759641764'],
        ['20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
        ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817864', '20231101_160337/1698825817964'],
        ['20231101_160337/1698825818064', '20231101_160337/1698825818164', '20231101_160337/1698825818264', '20231101_160337/1698825818364'],
    ]


def test_group_sampler_sample_val_groups_1scene(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = IndexGroupSampler("val", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_val_groups(_scene_frame_inds) == [
        [ "20231101_160337/1698825817664", "20231101_160337/1698825817764", "20231101_160337/1698825817864", "20231101_160337/1698825817964", ],
        [ "20231101_160337/1698825818064", "20231101_160337/1698825818164", "20231101_160337/1698825818264", "20231101_160337/1698825818364", ],
        [ "20231101_160337/1698825818464", "20231101_160337/1698825818564", "20231101_160337/1698825818664", "20231101_160337/1698825818764", ],
        [ "20231101_160337/1698825818864", "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", ],
        [ "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", "20231101_160337/1698825819264", ],
    ]

def test_group_sampler_sample_val_groups_frm_intvl1_simple():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = IndexGroupSampler("val", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_val_groups(_scene_frame_inds) == [
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03'], 
        ['Scn/04', 'Scn/05', 'Scn/06', 'Scn/07'], 
        ['Scn/08', 'Scn/09', 'Scn/10', 'Scn/11'], 
        ['Scn/12', 'Scn/13', 'Scn/14', 'Scn/15'],
        ['Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'],
    ]

def test_group_sampler_sample_val_groups_frm_intvl2_simple():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = IndexGroupSampler("val", possible_group_sizes=4, possible_frame_intervals=2, seed=42)
    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07'], 
        ['Scn/08', 'Scn/10', 'Scn/12', 'Scn/14'], 
        ['Scn/09', 'Scn/11', 'Scn/13', 'Scn/15'],
        ['Scn/10', 'Scn/12', 'Scn/14', 'Scn/16'],
    ]


def test_group_sampler_sample_val_groups_frm_intvl2_grp_intvl_just_fit():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(20)]}
    gbs = IndexGroupSampler("val", possible_group_sizes=10, possible_frame_intervals=2, seed=42)
    assert gbs.sample_val_groups(_scene_frame_inds) == [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06', 'Scn/08', 'Scn/10', 'Scn/12', 'Scn/14', 'Scn/16', 'Scn/18'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07', 'Scn/09', 'Scn/11', 'Scn/13', 'Scn/15', 'Scn/17', 'Scn/19'], 
    ]


def test_group_sampler_sample_val_groups_frm_intvl2(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = IndexGroupSampler("val", possible_group_sizes=4, possible_frame_intervals=2, seed=42)
    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
        ['20231101_160337/1698825817664', '20231101_160337/1698825817864', '20231101_160337/1698825818064', '20231101_160337/1698825818264'],
        ['20231101_160337/1698825817764', '20231101_160337/1698825817964', '20231101_160337/1698825818164', '20231101_160337/1698825818364'],
        ['20231101_160337/1698825818464', '20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064'],
        ['20231101_160337/1698825818564', '20231101_160337/1698825818764', '20231101_160337/1698825818964', '20231101_160337/1698825819164'],
        ['20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064', '20231101_160337/1698825819264'],
    ]


def test_group_sampler_sample_val_groups_frm_intvl2_grp_sz10(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = IndexGroupSampler("val", possible_group_sizes=10, possible_frame_intervals=2, seed=42)

    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
        ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817964', '20231101_160337/1698825818164', '20231101_160337/1698825818364', '20231101_160337/1698825818564', '20231101_160337/1698825818764', '20231101_160337/1698825818964', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
        ['20231101_160337/1698825817664', '20231101_160337/1698825817864', '20231101_160337/1698825818064', '20231101_160337/1698825818264', '20231101_160337/1698825818464', '20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064', '20231101_160337/1698825819264', '20231101_160337/1698825819264'],
    ]

def test_group_sampler_sample_scene_groups(scene_frame_inds):
    gbs = IndexGroupSampler("test_scene_by_scene", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_scene_groups(scene_frame_inds) == [
        [ "20231101_160337/1698825817664", "20231101_160337/1698825817764", "20231101_160337/1698825817864", "20231101_160337/1698825817964", "20231101_160337/1698825818064", "20231101_160337/1698825818164", "20231101_160337/1698825818264", "20231101_160337/1698825818364", "20231101_160337/1698825818464", "20231101_160337/1698825818564", "20231101_160337/1698825818664", "20231101_160337/1698825818764", "20231101_160337/1698825818864", "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", "20231101_160337/1698825819264", ],
        [ "20231101_160337_subset/1698825818164", "20231101_160337_subset/1698825818264", "20231101_160337_subset/1698825818364", "20231101_160337_subset/1698825818464", "20231101_160337_subset/1698825818564", "20231101_160337_subset/1698825818664", "20231101_160337_subset/1698825818764", "20231101_160337_subset/1698825818864", ],
        [ "20230823_110018/1692759640764", "20230823_110018/1692759640864", "20230823_110018/1692759640964", "20230823_110018/1692759641064", "20230823_110018/1692759641164", "20230823_110018/1692759641264", "20230823_110018/1692759641364", "20230823_110018/1692759641464", "20230823_110018/1692759641564", "20230823_110018/1692759641664", "20230823_110018/1692759641764", ]
    ]

def test_group_sampler_convert_groups_to_info():
    groups = [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07'], 
        ['Scn/08', 'Scn/10', 'Scn/12', 'Scn/14'], 
        ['Scn/09', 'Scn/11', 'Scn/13', 'Scn/15'],
    ]
    groups_as_index_info = IndexGroupSampler._convert_groups_to_info(groups)
    def ii(scene_frame_str, prev=None, next=None):
        return IndexInfo.from_str(scene_frame_str, prev=prev, next=next)

    assert groups_as_index_info == [
        [ii('Scn/00', next=ii('Scn/02')), ii('Scn/02', prev=ii('Scn/00'), next=ii('Scn/04')), ii('Scn/04', prev=ii('Scn/02', next=ii('Scn/06'))), ii('Scn/06', prev=ii('Scn/04'))], 
        [ii('Scn/01', next=ii('Scn/03')), ii('Scn/03', prev=ii('Scn/01'), next=ii('Scn/05')), ii('Scn/05', prev=ii('Scn/03', next=ii('Scn/07'))), ii('Scn/07', prev=ii('Scn/05'))], 
        [ii('Scn/08', next=ii('Scn/10')), ii('Scn/10', prev=ii('Scn/08'), next=ii('Scn/12')), ii('Scn/12', prev=ii('Scn/10'), next=ii('Scn/14')), ii('Scn/14', prev=ii('Scn/12'))], 
        [ii('Scn/09', next=ii('Scn/11')), ii('Scn/11', prev=ii('Scn/09'), next=ii('Scn/13')), ii('Scn/13', prev=ii('Scn/11'), next=ii('Scn/15')), ii('Scn/15', prev=ii('Scn/13'))],
    ]

    
def test_cur_train_group_size():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = IndexGroupSampler("train", possible_group_sizes=[2, 4, 8], possible_frame_intervals=1, seed=52)
    assert gbs.sample_train_groups(_scene_frame_inds) == [
        ['Scn/10', 'Scn/11', 'Scn/12', 'Scn/13'], 
        ['Scn/02', 'Scn/03', 'Scn/04', 'Scn/05'], 
        ['Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03'], 
        ['Scn/06', 'Scn/07', 'Scn/08', 'Scn/09'], 
    ]
    assert gbs.group_size == 4
    gbs.seed = 42
    assert gbs.sample_train_groups(_scene_frame_inds) == [
        ['Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07', 'Scn/08'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07'], 
        ['Scn/09', 'Scn/10', 'Scn/11', 'Scn/12', 'Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
    ]
    assert gbs.group_size == 8


import pytest

from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.dataset import GroupBatchDataset
from prefusion.dataset.group_sampler import generate_groups, IndexGroupSampler



@pytest.fixture
def info_pkl_path():
    return {
        'scene-0001': {'frame_info': {
            "101": {
                '3d_boxes': {

                }, 
                '3d_polylines': {
                    
                }
            }
        }},
        'scene-0004': {'frame_info': {
            
        }},
    }


def test_generate_cbgs_groups(info_pkl_path):
    pass
    # gbs = GroupSampler(scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    # train_groups = gbs.sample_train_groups()
    # assert train_groups == [
    #     ['20230823_110018/1692759640764', '20230823_110018/1692759640864', '20230823_110018/1692759640964', '20230823_110018/1692759641064'],
    #     ['20231101_160337/1698825818864', '20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164'],
    #     ['20231101_160337/1698825818464', '20231101_160337/1698825818564', '20231101_160337/1698825818664', '20231101_160337/1698825818764'],
    #     ['20230823_110018/1692759641164', '20230823_110018/1692759641264', '20230823_110018/1692759641364', '20230823_110018/1692759641464'],
    #     ['20231101_160337_subset/1698825818164', '20231101_160337_subset/1698825818264', '20231101_160337_subset/1698825818364', '20231101_160337_subset/1698825818464'],
    #     ['20231101_160337_subset/1698825818564', '20231101_160337_subset/1698825818664', '20231101_160337_subset/1698825818764', '20231101_160337_subset/1698825818864'],
    #     ['20230823_110018/1692759641464', '20230823_110018/1692759641564', '20230823_110018/1692759641664', '20230823_110018/1692759641764'],
    #     ['20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
    #     ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817864', '20231101_160337/1698825817964'],
    #     ['20231101_160337/1698825818064', '20231101_160337/1698825818164', '20231101_160337/1698825818264', '20231101_160337/1698825818364'],
    # ]
