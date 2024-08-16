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
