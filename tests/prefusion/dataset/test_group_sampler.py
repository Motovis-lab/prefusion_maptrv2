import pickle
from typing import Dict, List
from collections import defaultdict
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import pytest

from copious.io.fs import mktmpdir, parent_ensured_path
from prefusion.dataset.utils import PolarDict
from prefusion.dataset.index_info import IndexInfo
from prefusion.dataset.group_sampler import (
    IndexGroupSampler, 
    SequentialSceneFrameGroupSampler,
    ClassBalancedGroupSampler, 
    generate_groups, 
    convert_str_index_to_index_info, 
    get_scene_frame_inds,
)


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


def _to_scene_frm_ids(groups: List[List[IndexInfo]]) -> List[List[str]]:
    return [[itm.scene_frame_id for itm in grp] for grp in groups]


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
    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
        [ "20231101_160337/1698825817664", "20231101_160337/1698825817764", "20231101_160337/1698825817864", "20231101_160337/1698825817964", ],
        [ "20231101_160337/1698825818064", "20231101_160337/1698825818164", "20231101_160337/1698825818264", "20231101_160337/1698825818364", ],
        [ "20231101_160337/1698825818464", "20231101_160337/1698825818564", "20231101_160337/1698825818664", "20231101_160337/1698825818764", ],
        [ "20231101_160337/1698825818864", "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", ],
        [ "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", "20231101_160337/1698825819264", ],
    ]

def test_group_sampler_sample_val_groups_frm_intvl1_simple():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = IndexGroupSampler("val", possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
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
    val_groups = gbs.sample_val_groups(_scene_frame_inds)
    assert val_groups == [
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
    gbs = SequentialSceneFrameGroupSampler("test_scene_by_scene", seed=42)
    info = defaultdict(lambda: defaultdict(dict))
    for scene in scene_frame_inds:
        for frm in scene_frame_inds[scene]:
            info[scene]["frame_info"][frm.split('/')[1]] = "dummy_directory/dummy_path.dummy"
    frame_info = PolarDict({scene_id: scene_data['frame_info'] for scene_id, scene_data in info.items()}, separator="/")
    scene_groups = gbs.sample(None, frame_info)
    assert _to_scene_frm_ids(scene_groups) == [
        ['20230823_110018/1692759640764'], ['20230823_110018/1692759640864'], ['20230823_110018/1692759640964'], ['20230823_110018/1692759641064'],
        ['20230823_110018/1692759641164'], ['20230823_110018/1692759641264'], ['20230823_110018/1692759641364'], ['20230823_110018/1692759641464'],
        ['20230823_110018/1692759641564'], ['20230823_110018/1692759641664'], ['20230823_110018/1692759641764'],
        ['20231101_160337/1698825817664'], ['20231101_160337/1698825817764'], ['20231101_160337/1698825817864'], ['20231101_160337/1698825817964'], 
        ['20231101_160337/1698825818064'], ['20231101_160337/1698825818164'], ['20231101_160337/1698825818264'], ['20231101_160337/1698825818364'], 
        ['20231101_160337/1698825818464'], ['20231101_160337/1698825818564'], ['20231101_160337/1698825818664'], ['20231101_160337/1698825818764'], 
        ['20231101_160337/1698825818864'], ['20231101_160337/1698825818964'], ['20231101_160337/1698825819064'], ['20231101_160337/1698825819164'], 
        ['20231101_160337/1698825819264'],
        ['20231101_160337_subset/1698825818164'], ['20231101_160337_subset/1698825818264'], ['20231101_160337_subset/1698825818364'], ['20231101_160337_subset/1698825818464'],
        ['20231101_160337_subset/1698825818564'], ['20231101_160337_subset/1698825818664'], ['20231101_160337_subset/1698825818764'], ['20231101_160337_subset/1698825818864'],
    ]

def test_group_sampler_convert_groups_to_info():
    groups = [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07'], 
        ['Scn/08', 'Scn/10', 'Scn/12', 'Scn/14'], 
        ['Scn/09', 'Scn/11', 'Scn/13', 'Scn/15'],
    ]
    groups_as_index_info = convert_str_index_to_index_info(groups)
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
    train_groups = gbs.sample_train_groups(_scene_frame_inds)
    assert train_groups == [
        ['Scn/10', 'Scn/11', 'Scn/12', 'Scn/13'], 
        ['Scn/02', 'Scn/03', 'Scn/04', 'Scn/05'], 
        ['Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03'], 
        ['Scn/06', 'Scn/07', 'Scn/08', 'Scn/09'], 
    ]
    assert gbs.group_size == 4
    gbs.seed = 42
    train_groups = gbs.sample_train_groups(_scene_frame_inds)
    assert train_groups == [
        ['Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07', 'Scn/08'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07'], 
        ['Scn/09', 'Scn/10', 'Scn/11', 'Scn/12', 'Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
    ]
    assert gbs.group_size == 8


@pytest.fixture
def mock_info():
    return PolarDict({
        "20230901_000000/1692759619664": {1},
        "20230901_000000/1692759619764": 1,
        "20231023_222222/1692759621364": [1],
        "20250428_000007/1722759001064": {6},
        "20250428_000007/1722759002064": [8],
        "20250428_000007/1722759000064": "1",
    })


def test_get_scene_frame_inds_with_no_indices_provided(mock_info):
    indices = get_scene_frame_inds(mock_info)
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664", "20230901_000000/1692759619764"],
        "20231023_222222": ["20231023_222222/1692759621364"],
        "20250428_000007": ["20250428_000007/1722759000064", "20250428_000007/1722759001064", "20250428_000007/1722759002064"],
    }


def test_get_scene_frame_inds_with_indices_provided(mock_info):
    indices = get_scene_frame_inds(mock_info, indices=["20230901_000000/1692759620664", "20231001_111111/1712759770664", "20230901_000000/1692759619664"])
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664"],
    }


def test_get_scene_frame_inds_with_duplicated_indices_provided(mock_info):
    indices = get_scene_frame_inds(mock_info, indices=["20230901_000000/1692759619664", "20230901_000000/1692759620664", "20231001_111111/1712759770664", "20230901_000000/1692759619664"])
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664", "20230901_000000/1692759619664"],
    }


dummy_sz = np.ones(3, dtype=np.float32).tolist()
dummy_rot = np.eye(3, dtype=np.float32)
dummy_pos = np.zeros(3, dtype=np.float32)
dummy_pts = np.arange(12, dtype=np.float32).reshape(-1, 3)

def generate_dummy_box(track_id: str, obj_type: str, door_open: str = None, barrier_type: str = None):
    bx = {'track_id': track_id, 'class': obj_type, 'size': dummy_sz, 'rotation': dummy_rot, 'translation': dummy_pos, 'attr': {}}
    if obj_type in ["car", "bus", "truck"] and door_open is not None:
        bx['attr'] = {"door_open": door_open}
    if obj_type == "barrier" and barrier_type is not None:
        bx['attr'] = {"barrier_type": barrier_type}
    return bx


def generate_dummy_polyline(track_id: str, obj_type: str, color='color.yellow'):
    return {'track_id': track_id, 'class': obj_type, 'points': dummy_pts, 'attr': {'color': color}}


def generate_dummy_boxes(prefix: str, cfg: Dict[str, int], door_open: str = None, barrier_type: str = None) -> List[Dict]:
    return [generate_dummy_box(f"{prefix}_{t}_{i}", obj_type, door_open=door_open, barrier_type=barrier_type) 
            for t, (obj_type, num) in enumerate(cfg.items()) for i in range(num)]


def generate_dummy_polylines(prefix: str, cfg: Dict[str, int], color='color.yellow') -> List[Dict]:
    return [generate_dummy_polyline(f"{prefix}_{t}_{i}", obj_type, color=color) 
            for t, (obj_type, num) in enumerate(cfg.items()) for i in range(num)]


@pytest.fixture
def dataset_info_pkl():
    return {
        'scene-0001': {'frame_info': {
            "101": {
                '3d_boxes': generate_dummy_boxes("101b", dict(barrier=1, car=2, truck=1, bicycle=1), barrier_type="barrier_type.soft"),
                '3d_polylines': generate_dummy_polylines("101p", dict(parking_slot=1, laneline=1), color='color.yellow'),
            },
            "102": {
                '3d_boxes': generate_dummy_boxes("102b", dict(barrier=1, car=3, truck=2, bicycle=1, pedestrian=1), barrier_type="barrier_type.hard"),
                '3d_polylines': {},
            },
            "103": {
                '3d_boxes': {},
                '3d_polylines': generate_dummy_polylines("103p", dict(laneline=1), color='color.yellow'),
            },
            "104": {
                '3d_boxes': generate_dummy_boxes("104b", dict(pedestrian=1, barrier=2, truck=1, car=2), barrier_type="barrier_type.hard"),
                '3d_polylines': generate_dummy_polylines("104p", dict(parking_slot=3), color='color.white'),
            },
            "105": {
                '3d_boxes': generate_dummy_boxes("105b", dict(pedestrian=1, barrier=2, truck=1, bicycle=1), door_open="door_open.True", barrier_type="barrier_type.hard"),
                '3d_polylines': generate_dummy_polylines("105p", dict(parking_slot=1, laneline=1), color='color.yellow'),
            },
            "106": {
                '3d_boxes': generate_dummy_boxes("106b", dict(car=5, truck=2)),
                '3d_polylines': generate_dummy_polylines("106p", dict(laneline=1), color='color.white'),
            },
        }, 'scene_info': {'attr': {
            'attr.environment.light': 'attr.environment.light.natural_light',
            'attr.area.parking_lot.type': 'attr.area.parking_lot.type.outdoor'}
        }},

        'scene-0004': {'frame_info': {
            "401": {
                '3d_boxes': generate_dummy_boxes("401b", dict(car=2, bus=1, bicycle=2)),
                '3d_polylines': generate_dummy_polylines("401p", dict(parking_slot=1, laneline=1), color='color.yellow'),
            },
            "402": {
                '3d_boxes': generate_dummy_boxes("402b", dict(car=3, truck=1, bicycle=1, pedestrian=1)),
                '3d_polylines': {},
            },
            "403": {
                '3d_boxes': {},
                '3d_polylines': generate_dummy_polylines("403p", dict(access_aisle=1), color='color.yellow'),
            },
            "404": {
                '3d_boxes': generate_dummy_boxes("404b", dict(pedestrian=1, car=2, bus=1)),
                '3d_polylines': generate_dummy_polylines("404p", dict(parking_slot=1), color='color.yellow'),
            },
            "405": {
                '3d_boxes': generate_dummy_boxes("405b", dict(pedestrian=1, barrier=1, bus=1, bicycle=1), door_open="door_open.True", barrier_type="barrier_type.soft"),
                '3d_polylines': generate_dummy_polylines("405p", dict(laneline=2), color='color.yellow'),
            },
            "406": {
                '3d_boxes': {},
                '3d_polylines': generate_dummy_polylines("406p", dict(parking_slot=2), color='color.white'),
            },
            "407": {
                '3d_boxes': {},
                '3d_polylines': generate_dummy_polylines("407p", dict(parking_slot=2, laneline=2), color='color.white'),
            },
            "408": {
                '3d_boxes': {},
                '3d_polylines': {},
            },
            "409": {
                '3d_boxes': generate_dummy_boxes("408b", dict(car=3, bus=1, bicycle=1, pedestrian=1)),
                '3d_polylines': {},
            },
            "410": {
                '3d_boxes': generate_dummy_boxes("410b", dict(car=1, bus=2, bicycle=2), door_open="door_open.False"),
                '3d_polylines': {},
            },
            "411": {
                '3d_boxes': generate_dummy_boxes("411b", dict(pedestrian=2, car=4, bicycle=1)),
                '3d_polylines': {},
            },
            "412": {
                '3d_boxes': generate_dummy_boxes("412b", dict(car=3, bus=1)),
                '3d_polylines': {},
            },
        }, 'scene_info': {'attr': {
            'attr.environment.light': 'attr.environment.light.artificial_light',
            'attr.area.parking_lot.type': 'attr.area.parking_lot.type.outdoor'}
        }},

        'scene-0005': {'frame_info': {
            "501": {
                '3d_boxes': generate_dummy_boxes("501b", dict(car=1)),
                '3d_polylines': generate_dummy_polylines("501p", dict(parking_slot=1), color='color.blue'),
            },
        }, 'scene_info': {'attr': {
            'attr.environment.light': 'attr.environment.light.artificial_light',
            'attr.area.parking_lot.type': 'attr.area.parking_lot.type.indoor'}
        }},
    }


def get_separated_scene_info_and_frame_info(info: Dict, data_root: Path):
    scene_dict = {}
    for scene_id, scene_data in info.items():
        frame_info = scene_data.get('frame_info', {})
        for frame_id in frame_info.keys():
            frame_pkl_save_path = parent_ensured_path(data_root / scene_id / "frame_info_pkl" / f"{frame_id}.pkl")

            # Save the frame pickle data to the constructed path
            with frame_pkl_save_path.open('wb') as f_out:
                pickle.dump(frame_info[frame_id], f_out)

            # Update the frame_data to reference the saved file path
            frame_info[frame_id] = str(frame_pkl_save_path.relative_to(data_root))
        
        scene_pkl_save_path = parent_ensured_path(data_root / scene_id / "scene_info.pkl")
        with scene_pkl_save_path.open('wb') as f_out:
            pickle.dump(scene_data["scene_info"], f_out)
        
        scene_dict[scene_id] = str(scene_pkl_save_path.relative_to(data_root))

    scene_info = PolarDict(scene_dict)
    frame_info = PolarDict({scene_id: scene_data['frame_info'] for scene_id, scene_data in info.items()})
    return scene_info, frame_info


@pytest.fixture
def transformable_cfg():
    return dict(
        bbox_3d=dict(
            type='Bbox3D',
            loader=dict(
                type="AdvancedBbox3DLoader",
                class_mapping=dict(
                    car=["car"],
                    bus=["bus"],
                    truck=["truck"],
                    pedestrian=["pedestrian"],
                    bicycle=["bicycle"],
                    barrier_soft=['barrier::barrier_type.soft'],
                    barrier_hard=['barrier::barrier_type.hard'],
                ),
                attr_mapping=dict(
                    door_open=["door_open.True"],
                ),
            ),
        ),
        polyline_3d=dict(
            type='Polyline3D',
            dictionary=dict(
                classes=['parking_slot', 'laneline', 'access_aisle'],
                attr=['color'],
            ),
        ),
        camera_images=dict(
            type='CameraImageSet',
            loader=dict(type="NuscenesCameraImageSetLoader"),
            tensor_smith=dict(type='CameraImageTensor')
        ),
    )


@pytest.fixture
def cbgs_cfg():
    return {
        'desired_ratio': 0.5,
        'counter_type': 'frame', # choices: ['group', 'frame', 'object']
        # 'reference': 'the_max',
        'update_stats_during_oversampling': False,
        'oversampling_consider_no_objects': True,
        'oversampling_consider_object_attr': True,
    }


def test_index_group_sampler(dataset_info_pkl):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    gbs = IndexGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42)
    groups = gbs.sample(tmpdir, frame_info, phase='train')
    assert [[f.frame_id for f in group] for group in groups] == [
        ['401', '402', '403'], # ├─ car=5, bus=1, truck=1, pedestrian=1, bicycle=3
                               # ├─ parking_slot=1, laneline=1, access_aisle=1
                               # └─ color.yellow=3

        ['409', '410', '411'], # ├─ car=8, bus=3, pedestrian=3, bicycle=4

        ['410', '411', '412'], # ├─ car=8, bus=3, pedestrian=2, bicycle=3

        ['403', '404', '405'], # ├─ car=2, bus=2, pedestrian=2, bicycle=1, barrier_soft=1
                               # ├─ parking_slot=1, laneline=2, access_aisle=1
                               # └─ color.yellow=4, door_open=1

        ['501', '501', '501'], # ├─ car=3, parking_slot=3, color.blue=3

        ['104', '105', '106'], # ├─ car=7, truck=4, pedestrian=2, bicycle=1, barrier_hard=4
                               # ├─ parking_slot=4, laneline=2
                               # └─ color.yellow=2, color.white=4, door_open=1

        ['406', '407', '408'], # ├─ parking_slot=4, laneline=2
                               # ├─ color.white=6
                               # └─ <NO_OBJECTS>=1

        ['101', '102', '103'], # ├─ car=5, truck=3, pedestrian=1, bicycle=2, barrier_hard=1, barrier_soft=1
                               # ├─ parking_slot=1, laneline=2
                               # └─ color.yellow=3

        ['103', '104', '105']] # ├─ car=2, truck=2, pedestrian=2, bicycle=1, barrier_hard=4
                               # ├─ parking_slot=4, laneline=2
                               # └─ color.yellow=3, color.white=3, door_open=1


def test_cgbs_count_frame_level_class_occurrence(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': True, 'oversampling_consider_object_attr': True, 'counter_type': 'group'}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    assert len(groups) == 9
    assert groups[0].obj_cnt == {'car': 5, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 3, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 3}
    assert groups[1].obj_cnt == {'car': 8, 'bus': 3, 'pedestrian': 3, 'bicycle': 4}
    assert groups[2].obj_cnt == {'car': 8, 'bus': 3, 'pedestrian': 2, 'bicycle': 3}
    assert groups[3].obj_cnt == {'car': 2, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2, 'access_aisle': 1, 'color.yellow': 4, 'door_open': 1}
    assert groups[4].obj_cnt == {'car': 3, 'parking_slot': 3, 'color.blue': 3}
    assert groups[5].obj_cnt == {'car': 7, 'truck': 4, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 4, 'parking_slot': 4, 'laneline': 2, 'color.yellow': 2, 'color.white': 4, 'door_open': 1}
    assert groups[6].obj_cnt == {'parking_slot': 4, 'laneline': 2, 'color.white': 6, '<NO_OBJECTS>': 1}
    assert groups[7].obj_cnt == {'car': 5, 'truck': 3, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2, 'color.yellow': 3}
    assert groups[8].obj_cnt == {'car': 2, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 4, 'parking_slot': 4, 'laneline': 2, 'color.yellow': 3, 'color.white': 3, 'door_open': 1}

    assert groups[0].frm_cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 2}
    assert groups[1].frm_cnt == {'car': 3, 'bus': 2, 'pedestrian': 2, 'bicycle': 3}
    assert groups[2].frm_cnt == {'car': 3, 'bus': 2, 'pedestrian': 1, 'bicycle': 2}
    assert groups[3].frm_cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 3, 'door_open': 1}
    assert groups[4].frm_cnt == {'car': 3, 'parking_slot': 3, 'color.blue': 3}
    assert groups[5].frm_cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2, 'color.yellow': 1, 'color.white': 2, 'door_open': 1}
    assert groups[6].frm_cnt == {'parking_slot': 2, 'laneline': 1, 'color.white': 2, '<NO_OBJECTS>': 1}
    assert groups[7].frm_cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2, 'color.yellow': 2}
    assert groups[8].frm_cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2, 'color.yellow': 2, 'color.white': 1, 'door_open': 1}

    assert groups[0].grp_cnt == {'car': 1, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 1}
    assert groups[1].grp_cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1}
    assert groups[2].grp_cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1}
    assert groups[3].grp_cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 1, 'door_open': 1}
    assert groups[4].grp_cnt == {'car': 1, 'parking_slot': 1, 'color.blue': 1}
    assert groups[5].grp_cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1, 'color.white': 1, 'door_open': 1}
    assert groups[6].grp_cnt == {'parking_slot': 1, 'laneline': 1, 'color.white': 1, '<NO_OBJECTS>': 1}
    assert groups[7].grp_cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1}
    assert groups[8].grp_cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1, 'color.white': 1, 'door_open': 1}

    assert groups[0].cnt == {'car': 1, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 1}
    assert groups[1].cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1}
    assert groups[2].cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1}
    assert groups[3].cnt == {'car': 1, 'bus': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1, 'color.yellow': 1, 'door_open': 1}
    assert groups[4].cnt == {'car': 1, 'parking_slot': 1, 'color.blue': 1}
    assert groups[5].cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1, 'color.white': 1, 'door_open': 1}
    assert groups[6].cnt == {'parking_slot': 1, 'laneline': 1, 'color.white': 1, '<NO_OBJECTS>': 1}
    assert groups[7].cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1}
    assert groups[8].cnt == {'car': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 1, 'barrier_hard': 1, 'parking_slot': 1, 'laneline': 1, 'color.yellow': 1, 'color.white': 1, 'door_open': 1}


def test_oversample_classes_1(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False, "counter_type": "frame"}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.sample_minority_groups(groups, ["barrier_soft", "access_aisle"], "barrier_hard", target_ratio=1.0)
    assert len(sampled_groups) == 6
    assert sampled_groups[0].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[1].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2}
    assert sampled_groups[2].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[3].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[4].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[5].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}


def test_oversample_classes_2(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False, "counter_type": "frame"}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.sample_minority_groups(groups, ["barrier_soft", "access_aisle", "barrier_hard"], "car", target_ratio=0.3)
    assert len(sampled_groups) == 3
    assert sampled_groups[0].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[1].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[2].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}


def _calc_class_distribution(groups):
    groups_df = pd.DataFrame([grp.cnt for grp in groups]).fillna(0.0)
    cnt_per_class = groups_df.sum(axis=0).to_frame(name="cnt")
    max_class = cnt_per_class.cnt.idxmax()
    cnt_per_class.loc[:, "ratio"] = cnt_per_class.cnt / cnt_per_class.loc[max_class].cnt
    return cnt_per_class


def test_oversample_0(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.1, "update_stats_during_oversampling": False}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.iterative_sample_minority_groups(groups)
    assert len(sampled_groups) == 0


def test_oversample_1(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.3, "update_stats_during_oversampling": False}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.iterative_sample_minority_groups(groups)
    assert len(sampled_groups) == 9
    assert sampled_groups[0].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[1].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[2].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[3].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[4].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2}
    assert sampled_groups[5].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[6].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[7].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[8].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}


def test_oversample_2(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.5, "update_stats_during_oversampling": False}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.iterative_sample_minority_groups(groups)
    assert len(sampled_groups) == 21
    # [access_aisle, barrier_soft] => barrier_hard
    assert sampled_groups[0].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[1].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[2].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[3].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[4].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2}
    assert sampled_groups[5].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}

    # [access_aisle, barrier_soft, barrier_hard] => bus
    assert sampled_groups[6].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[7].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[8].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2}
    assert sampled_groups[9].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[10].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}
    assert sampled_groups[11].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}

    # [access_aisle, barrier_soft, barrier_hard, bus] => truck
    assert sampled_groups[12].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[13].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[14].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}
    assert sampled_groups[15].cnt == {'car': 3, 'bus': 2, 'pedestrian': 2, 'bicycle': 3}

    # [access_aisle, barrier_soft, barrier_hard, bus, truck] => car
    assert sampled_groups[16].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[17].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[18].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}
    assert sampled_groups[19].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[20].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}


def test_oversample_update_stats_1(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.3, "update_stats_during_oversampling": True}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.iterative_sample_minority_groups(groups)
    assert len(sampled_groups) == 9
    assert sampled_groups[0].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[1].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[2].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[3].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[4].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2}
    assert sampled_groups[5].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[6].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[7].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1}
    assert sampled_groups[8].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2}


def test_oversample_update_stats_2(dataset_info_pkl, transformable_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.5, "update_stats_during_oversampling": True}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs._base_group_sampler.sample(tmpdir, frame_info, phase='train')
    groups = gbs.count_class_and_attr_occurrence(tmpdir, frame_info, groups)
    sampled_groups = gbs.iterative_sample_minority_groups(groups)
    assert len(sampled_groups) == 62
    i = 0
    # [access_aisle, barrier_soft] => barrier_hard
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1

    # [access_aisle, barrier_soft, barrier_hard] => bus
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1

    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1

    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1

    # [access_aisle, barrier_soft, barrier_hard, bus] => truck
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1

    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1

    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1

    assert sampled_groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 2, 'bicycle': 3} ; i += 1
    assert sampled_groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 1, 'bicycle': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 2, 'bicycle': 3} ; i += 1
    assert sampled_groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 1, 'bicycle': 2} ; i += 1
    assert sampled_groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert sampled_groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1


def test_generate_cbgs_groups(dataset_info_pkl, transformable_cfg, cbgs_cfg):
    tmpdir = mktmpdir()
    scene_info, frame_info = get_separated_scene_info_and_frame_info(dataset_info_pkl, tmpdir)
    cbgs_cfg = {'oversampling_consider_no_objects': False, 'oversampling_consider_object_attr': False,  "counter_type": "frame", "desired_ratio": 0.3, "update_stats_during_oversampling": False}
    gbs = ClassBalancedGroupSampler("train", possible_group_sizes=3, possible_frame_intervals=1, seed=42, transformable_cfg=transformable_cfg, cbgs_cfg=cbgs_cfg)
    groups = gbs.sample(tmpdir, frame_info, phase='train')
    
    assert len(groups) == 18
    i = 0
    assert groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 2, 'bicycle': 3} ; i += 1
    assert groups[i].cnt == {'car': 3, 'bus': 2, 'pedestrian': 1, 'bicycle': 2} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 3, 'parking_slot': 3} ; i += 1
    assert groups[i].cnt == {'car': 2, 'truck': 3, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert groups[i].cnt == {'parking_slot': 2, 'laneline': 1} ; i += 1
    assert groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
    assert groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 2, 'bus': 1, 'truck': 1, 'pedestrian': 1, 'bicycle': 2, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 2, 'truck': 2, 'pedestrian': 1, 'bicycle': 2, 'barrier_hard': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 2} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 1, 'bus': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_soft': 1, 'parking_slot': 1, 'laneline': 1, 'access_aisle': 1} ; i += 1
    assert groups[i].cnt == {'car': 1, 'truck': 2, 'pedestrian': 2, 'bicycle': 1, 'barrier_hard': 2, 'parking_slot': 2, 'laneline': 2} ; i += 1
