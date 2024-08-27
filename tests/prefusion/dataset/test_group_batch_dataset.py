from pathlib import Path

import pytest

from prefusion.dataset.dataset import GroupBatchDataset

@pytest.fixture
def mock_info():
    return {
        "20230901_000000": {"frame_info": {"1692759619664": {1}, "1692759619764": 1}},
        "20230823_111111": {"frame_info": {}},
        "20231023_222222": {"frame_info": {"1692759621364": [1]}},
    }

def test_prepare_indices_with_no_indices_provided(mock_info):
    indices = GroupBatchDataset._prepare_indices(mock_info)
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664", "20230901_000000/1692759619764"],
        "20231023_222222": ["20231023_222222/1692759621364"],
    }


def test_prepare_indices_with_indices_provided(mock_info):
    indices = GroupBatchDataset._prepare_indices(mock_info, ["20230901_000000/1692759619664"])
    assert indices == {
        "20230901_000000": ["20230901_000000/1692759619664"],
    }


class DummyTransform:
    def __init__(self, scope='frame') -> None: self.scope = scope
    def __call__(self, *transformables, **kwargs): return transformables


def test_sample_train_groups():
    dataset = GroupBatchDataset(
        name='gbd',
        data_root=Path('/Users/rlan/work/dataset/motovis/mv4d'),
        info_path=Path('/Users/rlan/work/dataset/motovis/mv4d/mv4d_infos.pkl'),
        transformable_keys=[
            'camera_images'
        ],
        dictionary={},
        tensor_smith={},
        transforms=[
            DummyTransform(scope='group')
        ],
        model_feeder={},
        phase='train',
        batch_size=1,
        group_size=2
    )
