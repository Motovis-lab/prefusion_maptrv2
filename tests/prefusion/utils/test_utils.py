import cv2
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal
import pytest
import torch
from copious.io.fs import mktmpdir
from pypcd_imp import pypcd

from prefusion.dataset.utils import read_pcd, make_seed, read_ego_mask, T4x4, get_reversed_mapping, unstack_batch_size


@pytest.fixture
def pcd_path():
    tmpdir = mktmpdir()
    pc_data = np.array(
        [(0, 0, 0, 18), (0.5, -1.5, 2.5, 50)], dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]
    )
    metadata = {
        'version': 0.7,
        'fields': ['x', 'y', 'z', 'intensity'],
        'size': [4, 4, 4, 4],
        'type': ['F', 'F', 'F', 'F'],
        'count': [1, 1, 1, 1],
        'width': 2,
        'height': 1,
        'viewpoint': [0, 0, 0, 1],
        'points': 2,
        'data': 'binary',
    }
    pcd = pypcd.PointCloud(metadata, pc_data)
    # pcd.pc_data['x'] = np.array([0, 0.5])
    # pcd.pc_data['y'] = np.array([0, -1.5])
    # pcd.pc_data['z'] = np.array([0, 2.5])
    # pcd.pc_data['intensity'] = np.array([18, 50])
    pcd_save_path = tmpdir / "pointcloud.pcd"
    pcd.save_pcd(str(pcd_save_path), compression='binary')
    return pcd_save_path


def test_read_pcd(pcd_path):
    points = read_pcd(str(pcd_path), intensity=True)
    assert points.shape == (2, 4)
    np.testing.assert_almost_equal(points[:, :3], np.array([[0, 0, 0], [0.5, -1.5, 2.5]]))
    np.testing.assert_almost_equal(points[:, 3], np.array([18, 50]))
    points = read_pcd(str(pcd_path), intensity=True)


def test_make_seed_1():
    seeds = [make_seed(3, i, base=13) for i in range(3)]
    assert seeds == [4, 5, 6]


def test_make_seed_2():
    seeds = [make_seed(2, i, j, base=10) for i in range(2) for j in range(3)]
    assert seeds == [13, 14, 15, 23, 24, 25]


def test_make_seed_3():
    seeds = [make_seed(2, i, j, base=10) for i in range(6) for j in range(2)]
    assert seeds == [13, 14, 23, 24, 33, 34, 43, 44, 53, 54, 63, 64]


def test_make_seed_4():
    seeds = [make_seed(2, i, j, k, base=10) for i in range(2) for j in range(2) for k in range(2)]
    assert seeds == [113, 114, 123, 124, 213, 214, 223, 224]


def test_read_ego_mask():
    tmpdir = mktmpdir()
    img = np.ones([2,4], dtype=np.uint8)
    save_path = str(tmpdir / "mask243.png")
    cv2.imwrite(save_path, img)
    assert_array_almost_equal(read_ego_mask(save_path), img)

    img255 = np.ones([2,4], dtype=np.uint8) * 255
    save_path = str(tmpdir / "mask255.png")
    cv2.imwrite(save_path, img255)
    assert_array_almost_equal(read_ego_mask(save_path),  img)


def test_t4x4():
    mat = T4x4(np.arange(9).reshape(3, 3), np.array([1, 2, 3]))
    assert_array_almost_equal(mat, np.array([[0, 1, 2, 1],
                                             [3, 4, 5, 2],
                                             [6, 7, 8, 3],
                                             [0, 0, 0, 1]]))


def test_get_reversed_mapping_1():
    assert get_reversed_mapping({"a": ["b", 1], 2: [18, "33"], "c": [77], 5: ["d"]}) == {
        "b": "a",
        1: "a",
        18: 2,
        "33": 2,
        77: "c",
        "d": 5,
    }


def test_get_reversed_mapping_2():
    assert get_reversed_mapping({"a": ["b", "1"], "2": ["18", "33"], "c": ["77"], "5": ["d"]}) == {
        "b": "a",
        "1": "a",
        "18": "2",
        "33": "2",
        "77": "c",
        "d": "5",
    }


def test_get_reversed_mapping_3():
    assert get_reversed_mapping({"new_class_name1": ["c1::attr1.True"], "new_class_name2": ["c2::attr1.True"]}) == { 
        "c1::attr1.True": "new_class_name1", 
        "c2::attr1.True": "new_class_name2", 
    }


def test_unstack_batch_size_1():
    batch_data = {"seg": torch.randn(2, 3, 24, 24), "reg": torch.randn(2, 4, 24, 24)}
    unstacked_data = unstack_batch_size(batch_data)
    assert_array_almost_equal(unstacked_data[0]["seg"], batch_data["seg"][0])
    assert_array_almost_equal(unstacked_data[1]["seg"], batch_data["seg"][1])
    assert_array_almost_equal(unstacked_data[0]["reg"], batch_data["reg"][0])
    assert_array_almost_equal(unstacked_data[1]["reg"], batch_data["reg"][1])


def test_unstack_batch_size_2():
    batch_data = {"seg": torch.randn(1, 3, 24), "reg": torch.randn(1, 4, 24)}
    unstacked_data = unstack_batch_size(batch_data)
    assert_array_almost_equal(unstacked_data[0]["seg"], batch_data["seg"][0])
    assert_array_almost_equal(unstacked_data[0]["reg"], batch_data["reg"][0])


def test_unstack_batch_size_3():
    batch_data = {"seg": torch.randn(2, 3), "reg": torch.randn(2, 4)}
    unstacked_data = unstack_batch_size(batch_data)
    assert_array_almost_equal(unstacked_data[0]["seg"], batch_data["seg"][0])
    assert_array_almost_equal(unstacked_data[1]["seg"], batch_data["seg"][1])
    assert_array_almost_equal(unstacked_data[0]["reg"], batch_data["reg"][0])
    assert_array_almost_equal(unstacked_data[1]["reg"], batch_data["reg"][1])


def test_unstack_batch_size_4():
    with pytest.raises(AssertionError):
        _ = unstack_batch_size(torch.randn(2, 3, 24, 24))
    with pytest.raises(AssertionError):
        _ = unstack_batch_size(torch.randn(2))
    with pytest.raises(AssertionError):
        _ = unstack_batch_size({"seg": torch.randn(6, 3, 24, 24), "reg": torch.randn(2, 4, 24, 24)})
