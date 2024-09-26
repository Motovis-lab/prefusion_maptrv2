import numpy as np
import pytest
from copious.io.fs import mktmpdir
from pypcd_imp import pypcd

from prefusion.dataset.utils import read_pcd, make_seed


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
    seeds = [make_seed(3, i, exp_base=13) for i in range(3)]
    assert seeds == [4, 5, 6]


def test_make_seed_2():
    seeds = [make_seed(2, i, j, exp_base=10) for i in range(2) for j in range(3)]
    assert seeds == [31, 32, 33, 41, 42, 43]
