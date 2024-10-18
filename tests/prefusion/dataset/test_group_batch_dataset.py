from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mmengine.config.config import ConfigDict

from prefusion.dataset.dataset import GroupBatchDataset, GroupSampler, IndexInfo, generate_groups
from prefusion.dataset.model_feeder import BaseModelFeeder


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


class DummyTransform:
    def __init__(self, scope="frame") -> None:
        self.scope = scope

    def __call__(self, *transformables, **kwargs):
        return transformables


class DummyImgTensorSmith:
    def __call__(self, transformable, **kwds: Any) -> Any:
        return {"img": transformable.img}

class DummyDepthTensorSmith:
    def __call__(self, transformable, **kwds: Any) -> Any:
        return {"depth": transformable.img}

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


def test_load_ego_poses():
    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path("tests/prefusion/dataset"),
        info_path=Path("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl"),
        transformables=dict(my_ego_poses=dict(type="ego_poses")),
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        phase="val",
        possible_frame_intervals=2,
        batch_size=2,
        possible_group_sizes=4,
    )

    index_info = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764"), next=IndexInfo("20231101_160337", "1698825817964"))
    ego_pose_set = dataset.load_ego_poses('ego_poses', index_info)
    assert len(ego_pose_set.transformables) == 3
    assert list(ego_pose_set.transformables.keys()) == ['-1', '0', '+1']
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"

    index_info2 = IndexInfo("20231101_160337", "1698825817864", prev=IndexInfo("20231101_160337", "1698825817764", prev=IndexInfo("20231101_160337", "1698825817664")), next=IndexInfo("20231101_160337", "1698825817964"))
    ego_pose_set = dataset.load_ego_poses('ego_poses', index_info2)
    assert len(ego_pose_set.transformables) == 4
    assert list(ego_pose_set.transformables.keys()) == ['-2', '-1', '0', '+1']
    assert ego_pose_set.transformables['-2'].timestamp == "1698825817664"
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"

    index_info3 = IndexInfo(
        prev=IndexInfo("20231101_160337", "1698825817764", prev=IndexInfo("20231101_160337", "1698825817664")),
        scene_id="20231101_160337",
        frame_id="1698825817864",
        next=IndexInfo("20231101_160337", "1698825817964", next=IndexInfo("20231101_160337", "1698825818064"))
    )
    ego_pose_set = dataset.load_ego_poses('ego_poses', index_info3)
    assert len(ego_pose_set.transformables) == 5
    assert list(ego_pose_set.transformables.keys()) == ['-2', '-1', '0', '+1', '+2']
    assert ego_pose_set.transformables['-2'].timestamp == "1698825817664"
    assert ego_pose_set.transformables['-1'].timestamp == "1698825817764"
    assert ego_pose_set.transformables['0'].timestamp == "1698825817864"
    assert ego_pose_set.transformables['+1'].timestamp == "1698825817964"
    assert ego_pose_set.transformables['+2'].timestamp == "1698825818064"


def test_load_camera_depth():
    import mmengine
    import mmcv
    data = mmengine.load("tests/prefusion/dataset/mv4d-infos-for-test-001.pkl")
    IMG_KEYS = [
        'camera1', 'camera11', 'camera12', 'camera13', 'camera15', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8'
        ]
    from copious.io.fs import mktmpdir
    tmpdir = mktmpdir()
    camera_image_depth = {}
    self_mask = {}
    for camera_t in IMG_KEYS:
        camera_image_depth.update({
            camera_t : f"{camera_t}_depth.npz"
        })
        self_mask.update({
            camera_t: f"{camera_t}_mask.png"
        })
        np.savez_compressed(str(tmpdir / f"{camera_t}_depth.npz"), depth=np.ones((5, 10)).astype(np.float16))
        mmcv.imwrite(np.ones((5, 10)), str(tmpdir / f"{camera_t}_mask.png"))

    data["20231101_160337"]["frame_info"]["1698825817864"].update({'camera_image_depth': camera_image_depth})
    data["20231101_160337"]['scene_info']['camera_mask'] =  self_mask
    
    depth_path = tmpdir / "mv4d-infos-for-test-depth.pkl"
    mmengine.dump(data, depth_path)

    dataset = GroupBatchDataset(
        name="gbd",
        data_root=Path(tmpdir),
        info_path=depth_path,
        transformables=[],
        transforms=[DummyTransform(scope="group")],
        model_feeder=BaseModelFeeder(),
        phase="val",
        possible_frame_intervals=2,
        batch_size=2,
        possible_group_sizes=4,
    )

    index_info = IndexInfo("20231101_160337", "1698825817864")
    camera_depth = dataset.load_camera_depths('camera_depths', index_info)
    assert len(camera_depth.transformables) == 12

    depth_fish_front = np.load(Path(tmpdir) / Path("camera1_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera1'].img == depth_fish_front)

    depth_fish_front = np.load(Path(tmpdir) / Path("camera2_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera2'].img == depth_fish_front)

    depth_fish_front = np.load(Path(tmpdir) / Path("camera6_depth.npz"))['depth'][..., None].astype(np.float32)
    assert np.all(camera_depth.transformables['camera6'].img == depth_fish_front)

    
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