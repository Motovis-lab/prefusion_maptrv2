import pytest
from pathlib import Path
from contrib.pretrain.datasets.dataset import PretrainDataset_FrontData


def test_pretrain_dataset_frontdata_load_data_list():
    dataset = PretrainDataset_FrontData(
        reduce_zero_label=True,
        data_root="./",
        ann_file="tests/contrib/pretrain/index.txt",
        pipeline=[],
        test_mode=False
    )
    data_list = dataset.load_data_list()
    assert len(data_list) == 2
    assert data_list[0]['img_id'] == "tests/contrib/pretrain/data/2024-07-29_15-17-22_0001"
    assert data_list[1]['img_id'] == "tests/contrib/pretrain/data/2024-07-29_15-17-22_0005"