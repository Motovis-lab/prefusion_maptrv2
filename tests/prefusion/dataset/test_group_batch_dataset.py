from pathlib import Path

import numpy as np
import pytest

from prefusion.dataset.dataset import GroupBatchDataset, GroupSampler, IndexInfo, SubEpochManager, generate_groups, EndOfAllSubEpochs
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import TRANSFORMABLE_LOADERS


@pytest.fixture
def mock_info():
    return {
        "20230901_000000": {"frame_info": {"1692759619664": {1}, "1692759619764": 1}},
        "20230823_111111": {"frame_info": {}},
        "20231023_222222": {"frame_info": {"1692759621364": [1]}},
    }


def test_prepare_indices_with_no_indices_provided(mock_info):
    indices = GroupBatchDataset._prepare_scene_frame_inds(mock_info)
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664", "20230901_000000/1692759619764"],
        "20231023_222222": ["20231023_222222/1692759621364"],
    }


def test_prepare_indices_with_indices_provided(mock_info):
    indices = GroupBatchDataset._prepare_scene_frame_inds(mock_info, ["20230901_000000/1692759619664"])
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664"],
    }


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
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    train_groups = gbs.sample_train_groups()
    assert train_groups == [ 
        ['20231101_160337/1698825818864', '20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164'],
        ['20231101_160337/1698825818064', '20231101_160337/1698825818164', '20231101_160337/1698825818264', '20231101_160337/1698825818364'],
        ['20231101_160337/1698825818464', '20231101_160337/1698825818564', '20231101_160337/1698825818664', '20231101_160337/1698825818764'],
        ['20231101_160337/1698825818964', '20231101_160337/1698825819064', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
        ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817864', '20231101_160337/1698825817964'],
    ]

def test_group_sampler_sample_train_groups_more_scenes(scene_frame_inds):
    gbs = GroupSampler(scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    train_groups = gbs.sample_train_groups()
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
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_val_groups() == [
        [ "20231101_160337/1698825817664", "20231101_160337/1698825817764", "20231101_160337/1698825817864", "20231101_160337/1698825817964", ],
        [ "20231101_160337/1698825818064", "20231101_160337/1698825818164", "20231101_160337/1698825818264", "20231101_160337/1698825818364", ],
        [ "20231101_160337/1698825818464", "20231101_160337/1698825818564", "20231101_160337/1698825818664", "20231101_160337/1698825818764", ],
        [ "20231101_160337/1698825818864", "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", ],
        [ "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", "20231101_160337/1698825819264", ],
    ]

def test_group_sampler_sample_val_groups_frm_intvl1_simple():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_val_groups() == [
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03'], 
        ['Scn/04', 'Scn/05', 'Scn/06', 'Scn/07'], 
        ['Scn/08', 'Scn/09', 'Scn/10', 'Scn/11'], 
        ['Scn/12', 'Scn/13', 'Scn/14', 'Scn/15'],
        ['Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'],
    ]

def test_group_sampler_sample_val_groups_frm_intvl2_simple():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(17)]}
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=2, seed=42)
    val_groups = gbs.sample_val_groups()
    assert val_groups == [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07'], 
        ['Scn/08', 'Scn/10', 'Scn/12', 'Scn/14'], 
        ['Scn/09', 'Scn/11', 'Scn/13', 'Scn/15'],
        ['Scn/10', 'Scn/12', 'Scn/14', 'Scn/16'],
    ]


def test_group_sampler_sample_val_groups_frm_intvl2_grp_intvl_just_fit():
    _scene_frame_inds = {"Scn": [f"Scn/{i:02}" for i in range(20)]}
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=10, possible_frame_intervals=2, seed=42)
    assert gbs.sample_val_groups() == [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06', 'Scn/08', 'Scn/10', 'Scn/12', 'Scn/14', 'Scn/16', 'Scn/18'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07', 'Scn/09', 'Scn/11', 'Scn/13', 'Scn/15', 'Scn/17', 'Scn/19'], 
    ]


def test_group_sampler_sample_val_groups_frm_intvl2(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=2, seed=42)
    val_groups = gbs.sample_val_groups()
    assert val_groups == [
        ['20231101_160337/1698825817664', '20231101_160337/1698825817864', '20231101_160337/1698825818064', '20231101_160337/1698825818264'],
        ['20231101_160337/1698825817764', '20231101_160337/1698825817964', '20231101_160337/1698825818164', '20231101_160337/1698825818364'],
        ['20231101_160337/1698825818464', '20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064'],
        ['20231101_160337/1698825818564', '20231101_160337/1698825818764', '20231101_160337/1698825818964', '20231101_160337/1698825819164'],
        ['20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064', '20231101_160337/1698825819264'],
    ]


def test_group_sampler_sample_val_groups_frm_intvl2_grp_sz10(scene_frame_inds):
    _scene_frame_inds = {sid: values for sid, values in scene_frame_inds.items() if sid == "20231101_160337"}
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=10, possible_frame_intervals=2, seed=42)

    val_groups = gbs.sample_val_groups()
    assert val_groups == [
        ['20231101_160337/1698825817664', '20231101_160337/1698825817764', '20231101_160337/1698825817964', '20231101_160337/1698825818164', '20231101_160337/1698825818364', '20231101_160337/1698825818564', '20231101_160337/1698825818764', '20231101_160337/1698825818964', '20231101_160337/1698825819164', '20231101_160337/1698825819264'],
        ['20231101_160337/1698825817664', '20231101_160337/1698825817864', '20231101_160337/1698825818064', '20231101_160337/1698825818264', '20231101_160337/1698825818464', '20231101_160337/1698825818664', '20231101_160337/1698825818864', '20231101_160337/1698825819064', '20231101_160337/1698825819264', '20231101_160337/1698825819264'],
    ]

def test_group_sampler_sample_scene_groups(scene_frame_inds):
    gbs = GroupSampler(scene_frame_inds, possible_group_sizes=4, possible_frame_intervals=1, seed=42)
    assert gbs.sample_scene_groups() == [
        [ "20231101_160337/1698825817664", "20231101_160337/1698825817764", "20231101_160337/1698825817864", "20231101_160337/1698825817964", "20231101_160337/1698825818064", "20231101_160337/1698825818164", "20231101_160337/1698825818264", "20231101_160337/1698825818364", "20231101_160337/1698825818464", "20231101_160337/1698825818564", "20231101_160337/1698825818664", "20231101_160337/1698825818764", "20231101_160337/1698825818864", "20231101_160337/1698825818964", "20231101_160337/1698825819064", "20231101_160337/1698825819164", "20231101_160337/1698825819264", ],
        [ "20231101_160337_subset/1698825818164", "20231101_160337_subset/1698825818264", "20231101_160337_subset/1698825818364", "20231101_160337_subset/1698825818464", "20231101_160337_subset/1698825818564", "20231101_160337_subset/1698825818664", "20231101_160337_subset/1698825818764", "20231101_160337_subset/1698825818864", ],
        [ "20230823_110018/1692759640764", "20230823_110018/1692759640864", "20230823_110018/1692759640964", "20230823_110018/1692759641064", "20230823_110018/1692759641164", "20230823_110018/1692759641264", "20230823_110018/1692759641364", "20230823_110018/1692759641464", "20230823_110018/1692759641564", "20230823_110018/1692759641664", "20230823_110018/1692759641764", ]
    ]


def test_index_info_basic():
    ii = IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'next': {'scene_id': 'Scn', 'frame_id': '128'}}
    assert ii.prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'prev': None, 'next': {'scene_id': 'Scn', 'frame_id': '127'}}
    assert ii.next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'prev': {'scene_id': 'Scn', 'frame_id': '127'}, 'next': None}

def test_index_info_modify():
    ii = IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'next': {'scene_id': 'Scn', 'frame_id': '128'}}
    ii.frame_id = '888'
    assert ii.prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'prev': None, 'next': {'scene_id': 'Scn', 'frame_id': '888'}}
    assert ii.next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'prev': {'scene_id': 'Scn', 'frame_id': '888'}, 'next': None}


def test_index_info_eq():
    assert IndexInfo('Scn', '127') == IndexInfo('Scn', '127')
    assert IndexInfo('Scn', '127') != IndexInfo('Scn', '333')
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '126', next=IndexInfo('Scn', '127')).next
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) != IndexInfo('Scn', '126', next=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127')).prev
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'))).prev
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')))

def test_group_sampler_convert_groups_to_info():
    groups = [
        ['Scn/00', 'Scn/02', 'Scn/04', 'Scn/06'], 
        ['Scn/01', 'Scn/03', 'Scn/05', 'Scn/07'], 
        ['Scn/08', 'Scn/10', 'Scn/12', 'Scn/14'], 
        ['Scn/09', 'Scn/11', 'Scn/13', 'Scn/15'],
    ]
    groups_as_index_info = GroupSampler._convert_groups_to_info(groups)
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
    gbs = GroupSampler(_scene_frame_inds, possible_group_sizes=[2, 4, 8], possible_frame_intervals=1, seed=52)
    assert gbs.sample_train_groups() == [
        ['Scn/10', 'Scn/11', 'Scn/12', 'Scn/13'], 
        ['Scn/02', 'Scn/03', 'Scn/04', 'Scn/05'], 
        ['Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03'], 
        ['Scn/06', 'Scn/07', 'Scn/08', 'Scn/09'], 
    ]
    assert gbs.group_size == 4
    gbs.seed = 42
    assert gbs.sample_train_groups() == [
        ['Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07', 'Scn/08'], 
        ['Scn/00', 'Scn/01', 'Scn/02', 'Scn/03', 'Scn/04', 'Scn/05', 'Scn/06', 'Scn/07'], 
        ['Scn/09', 'Scn/10', 'Scn/11', 'Scn/12', 'Scn/13', 'Scn/14', 'Scn/15', 'Scn/16'], 
    ]
    assert gbs.group_size == 8


def test_batch_groups_1():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=1, 
        groups=[1, 2, 3], 
        batch_size=2
    )
    answer = [3, 2]
    assert batched_groups == answer


def test_batch_groups_2():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=1, 
        groups=[1, 2, 3, 4], 
        batch_size=3
    )
    answer = [4, 3, 2]
    assert batched_groups == answer


def test_batch_groups_3():
    batched_groups = GroupBatchDataset._batch_groups(
        group_batch_ind=0, 
        groups=[1, 2, 3, 4], 
        batch_size=3
    )
    answer = [1, 2, 3]
    assert batched_groups == answer


class DummyTransform:
    def __init__(self, scope="frame") -> None:
        self.scope = scope

    def __call__(self, *transformables, **kwargs):
        return transformables


def test_load_all_transformables():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset/example_inputs"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl"),
        transformables=dict(
            my_camera_images=dict(type="CameraImageSet"),
            my_ego_poses=dict(type="EgoPoseSet")
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        phase="val",
        possible_frame_intervals=2,
        batch_size=2,
        possible_group_sizes=4,
    )

    all_transformables = dataset.load_all_transformables(index_info)

    camera_image_set = all_transformables["my_camera_images"]
    assert camera_image_set.transformables['camera1'].img.sum() == 1752500
    assert camera_image_set.transformables['camera8'].ego_mask.sum() == 1365268
    np.testing.assert_almost_equal(
        camera_image_set.transformables['camera11'].intrinsic, 
        np.array([967.5516, 516.1143, 469.18085, 468.7578, 0.05346, -0.00585, -0.000539, -0.000161]),
        decimal=4
    )

    ego_pose_set = all_transformables["my_ego_poses"]
    assert len(ego_pose_set.transformables) == 3
    assert list(ego_pose_set.transformables.keys()) == ['-1', '0', '+1']
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"



@TRANSFORMABLE_LOADERS.register_module()
class CustomizedEgoPoseLoader:
    def __init__(self, data_root): self.data_root = data_root

    def load(self, name: str, scene_data, index_info, **kwargs):
        scene = scene_data['frame_info']
        return scene[index_info.frame_id]["ego_pose"]["rotation"], scene[index_info.frame_id]["ego_pose"]["translation"]


def test_load_all_transformables_customized_loader():
    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset/example_inputs"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl"),
        transformables=dict(
            single_frame_ego_pose=dict(
                type="EgoPose",
                loader=dict(type="CustomizedEgoPoseLoader")
            )
        ),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        phase="val",
        possible_frame_intervals=2,
        batch_size=2,
        possible_group_sizes=4,
    )

    all_transformables = dataset.load_all_transformables(index_info)

    ego_pose = all_transformables["single_frame_ego_pose"]
    np.testing.assert_almost_equal(ego_pose[0], np.array(
        [[ 9.99999639e-01,  6.82115545e-04,  5.07486245e-04],
       [-6.78401337e-04,  9.99973246e-01, -7.28335824e-03],
       [-5.12440759e-04,  7.28301133e-03,  9.99973347e-01]]))
    np.testing.assert_almost_equal(ego_pose[1], np.array([-0.02978186,  0.7788203 ,  0.01499793]))


def test_subepoch_manager_drop_last_false_false():
    mgr = SubEpochManager(3, 2, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.init(13)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    assert mgr.cur_subepoch_idx == 0
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_false_true():
    mgr = SubEpochManager(3, 2, drop_last_group_batch=False, drop_last_subepoch=True)
    mgr.init(13)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    assert mgr.cur_subepoch_idx == 0
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_true_false():
    mgr = SubEpochManager(3, 2, drop_last_group_batch=True, drop_last_subepoch=False)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_true_true():
    mgr = SubEpochManager(3, 2, drop_last_group_batch=True, drop_last_subepoch=True)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_reset():
    mgr = SubEpochManager(5, 2, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.init(16)
    assert mgr.num_total_group_batches == 4
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(0) == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()
    mgr.reset(13)
    assert mgr.num_total_group_batches == 3
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.cur_subepoch_idx == 1
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_translate_index():
    mgr = SubEpochManager(2, 4, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.init(19)
    assert mgr.num_total_group_batches == 10
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 4
    assert mgr.translate_index(0) == 0
    assert mgr.translate_index(1) == 1
    assert mgr.translate_index(2) == 2
    assert mgr.translate_index(3) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 4
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 5
    assert mgr.translate_index(2) == 6
    assert mgr.translate_index(3) == 7
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 8
    assert mgr.translate_index(1) == 9
    assert mgr.translate_index(2) == 2
    assert mgr.translate_index(3) == 7
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_visited():
    mgr = SubEpochManager(3, 2, drop_last_group_batch=True, drop_last_subepoch=True, debug_mode=True)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.visited.todict() == {}
    assert mgr.translate_index(0) == 0
    assert set(mgr.visited.todict()) == {0}
    assert mgr.translate_index(1) == 1
    assert set(mgr.visited.todict()) == {0, 1}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(0) == 2
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert set(mgr.visited.todict()) == {0, 1, 2}
    assert mgr.translate_index(1) == 3
    assert set(mgr.visited.todict()) == {0, 1, 2, 3}
    
    with pytest.warns(UserWarning) as warning_info:
        mgr.reset(18)
    assert str(warning_info[0].message) == "Some group batches are not visited! (group_batch_index: {4, 5})"

    assert mgr.num_total_group_batches == 6
    assert mgr.num_subepochs == 3
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert mgr.visited.todict() == {}
    assert mgr.translate_index(0) == 0
    assert set(mgr.visited.todict()) == {0}
    assert mgr.translate_index(1) == 1
    assert set(mgr.visited.todict()) == {0, 1}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(1) == 3
    assert set(mgr.visited.todict()) == {0, 1, 3}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(0) == 4
    assert set(mgr.visited.todict()) == {0, 1, 3, 4}
    with pytest.warns(UserWarning) as warning_info:
        mgr.reset(6)
    assert str(warning_info[0].message) == "Some group batches are not visited! (group_batch_index: {2, 5})"
