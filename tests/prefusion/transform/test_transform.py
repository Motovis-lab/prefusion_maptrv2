import pytest
import numpy as np

import mmcv

from prefusion.dataset.transform import random_transform_class_factory
from prefusion.dataset.transform import CameraImage, CameraImageSet
from prefusion.dataset.transform import RenderIntrinsic, RenderExtrinsic, RandomRenderExtrinsic


class MockTransformable:
    def __init__(self, data: str): self.data = data
    def wrap(self, *, wrapper='()', **kwargs):
        self.data = wrapper[0] + self.data  + wrapper[-1]
        return self

def test_random_transform_class_factory():
    trb = MockTransformable("hello")
    RandomWrapTransform = random_transform_class_factory("RandomWrapTransform", "wrap")
    # transform = RandomWrapTransform(prob=1.0, param_randomization_rules={"wrap": })



@pytest.fixture()
def fisheye_image():
    return CameraImage(
        cam_id='VCAMERA_FISHEYE_FRONT',
        cam_type='FisheyeCamera', 
        img=mmcv.imread('./data/test_fisheye.jpg'), 
        ego_mask=mmcv.imread('./data/test_fisheye_ego_mask.png', flag='grayscale'), 
        extrinsic=(np.array([[ 1.38777878e-16, -5.00000000e-01,  8.66025404e-01],
                            [-1.00000000e+00, -8.32667268e-17,  1.66533454e-16],
                            [ 0.00000000e+00, -8.66025404e-01, -5.00000000e-01]]),
                np.array([3.81686209, 0.0055232 , 0.74317797])),
        intrinsic=[47.5, 31.5, 24, 24, 0.1, 0, 0, 0]
    )

@pytest.fixture()
def perspective_image():
    return CameraImage(
        cam_id='VCAMERA_PERSPECTIVE_FRONT',
        cam_type='PerspectiveCamera', 
        img=mmcv.imread('./data/test_perspective.jpg'), 
        ego_mask=mmcv.imread('./data/test_perspective_ego_mask.png', flag='grayscale'), 
        extrinsic=(np.array([[ 2.22044605e-16,  2.22044605e-16,  1.00000000e+00],
                            [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16],
                            [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]]),
                np.array([1.64080219, 0.05154541, 1.59953608])), 
        intrinsic=[63.5, 47.5, 64, 64]
    )


def test_render_intrinsic(fisheye_image, perspective_image):
    transform = RenderIntrinsic(
        resolutions={
            'VCAMERA_FISHEYE_FRONT': (96, 48),
            'VCAMERA_PERSPECTIVE_FRONT': (96, 48),
        }
    )
    transform(fisheye_image, perspective_image)
    answer_fisheye_img = mmcv.imread('./data/test_fisheye_render_intrinsic_result.jpg')
    answer_perspective_img = mmcv.imread('./data/test_perspective_render_intrinsic_result.jpg')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img, decimal=3)
    np.testing.assert_almost_equal(fisheye_image.intrinsic, [47.5, 23.5, 24.0, 24.0, 0.1, 0, 0, 0])
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img, decimal=3)
    np.testing.assert_almost_equal(perspective_image.intrinsic, [47.5, 23.5, 48.0, 48.0])


def test_render_extrinsic(fisheye_image, perspective_image):
    transform = RenderExtrinsic(
        del_rotations={
            'VCAMERA_FISHEYE_FRONT': [0, -30, 0],
            'VCAMERA_PERSPECTIVE_FRONT': [0, 0, 30]
        }
    )
    transform(fisheye_image, perspective_image)
    answer_fisheye_img = mmcv.imread('./data/test_fisheye_render_extrinsic_result.jpg')
    answer_perspective_img = mmcv.imread('./data/test_perspective_render_extrinsic_result.jpg')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img, decimal=3)
    np.testing.assert_almost_equal(
        fisheye_image.extrinsic[0], 
        np.array([[ 1.20185168e-16,  1.07780618e-10,  1.00000000e+00],
                  [-1.00000000e+00, -8.32667268e-17,  1.66533454e-16],
                  [ 6.93889390e-17, -1.00000000e+00,  1.07780618e-10]])
    )
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img, decimal=3)
    np.testing.assert_almost_equal(
        perspective_image.extrinsic[0], 
        np.array([[ 5.00000000e-01,  1.92296269e-16,  8.66025404e-01],
                  [-8.66025404e-01,  1.11022302e-16,  5.00000000e-01],
                  [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]])
    )
