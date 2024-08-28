import random
from typing import Tuple

import pytest
import numpy as np

import mmcv

from prefusion.dataset.transform import (
    random_transform_class_factory,
    CameraImage, CameraImageSet,
    RandomSetIntrinsicParam, RandomSetExtrinsicParam,
    RenderIntrinsic, RenderExtrinsic, RandomRenderExtrinsic,
)

class MockTextTransformable:
    def __init__(self, text: str):
        self.text = text
        self.number = len(text)

    def wrap(self, *, wrapper="()", **kwargs):
        self.text = wrapper[0] + self.text + wrapper[-1]
        return self

    def add_then_divide(self, *, delta: int, divisor: float, **kwargs):
        self.number += delta
        self.number /= divisor
        return self


def test_random_transform_class_factory_enum():
    txt = MockTextTransformable("hello")
    RandomWrapTransform = random_transform_class_factory("RandomWrapTransform", "wrap")
    transform = RandomWrapTransform(
        prob=1.0, param_randomization_rules={"wrapper": {"type": "enum", "choices": ["[]", "()", "<>"]}}, scope="frame"
    )
    assert txt.text == "hello"
    _ = transform(txt, seeds={"frame": 42})
    assert txt.text == "<hello>"
    _ = transform(txt, seeds={"frame": 43})
    assert txt.text == "[<hello>]"
    _ = transform(txt, seeds={"frame": 44})
    assert txt.text == "([<hello>])"



@pytest.fixture()
def fisheye_image():
    return CameraImage(
        cam_id='VCAMERA_FISHEYE_FRONT',
        cam_type='FisheyeCamera', 
        img=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye.jpg'), 
        ego_mask=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_ego_mask.png', flag='grayscale'), 
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
        img=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective.jpg'), 
        ego_mask=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_ego_mask.png', flag='grayscale'), 
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
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_intrinsic_result.jpg')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_intrinsic_result.jpg')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img, decimal=2)
    np.testing.assert_almost_equal(fisheye_image.intrinsic, [47.5, 23.5, 24.0, 24.0, 0.1, 0, 0, 0])
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img, decimal=2)
    np.testing.assert_almost_equal(perspective_image.intrinsic, [47.5, 23.5, 48.0, 48.0])


def test_render_extrinsic(fisheye_image, perspective_image):
    transform = RenderExtrinsic(
        del_rotations={
            'VCAMERA_FISHEYE_FRONT': [0, -30, 0],
            'VCAMERA_PERSPECTIVE_FRONT': [0, 0, 30]
        }
    )
    transform(fisheye_image, perspective_image)
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_extrinsic_result.jpg')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_extrinsic_result.jpg')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img, decimal=2)
    np.testing.assert_almost_equal(
        fisheye_image.extrinsic[0], 
        np.array([[ 1.20185168e-16,  1.07780618e-10,  1.00000000e+00],
                  [-1.00000000e+00, -8.32667268e-17,  1.66533454e-16],
                  [ 6.93889390e-17, -1.00000000e+00,  1.07780618e-10]])
    )
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img, decimal=2)
    np.testing.assert_almost_equal(
        perspective_image.extrinsic[0], 
        np.array([[ 5.00000000e-01,  1.92296269e-16,  8.66025404e-01],
                  [-8.66025404e-01,  1.11022302e-16,  5.00000000e-01],
                  [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]])
    )
def test_random_transform_class_factory_float_int():
    txt = MockTextTransformable("hello")
    RandomAddThenDivideTransform = random_transform_class_factory("RandomAddDivideTransform", "add_then_divide")
    transform = RandomAddThenDivideTransform(
        prob=1.0,
        param_randomization_rules={
            "delta": {"type": "int", "range": [-2, 2]},
            "divisor": {"type": "float", "range": [1.0, 2.0]},
        },
        scope="batch",
    )
    assert txt.text == "hello"
    assert txt.number == 5
    _ = transform(txt, seeds={"frame": 42, "batch": 142, "group": 1142})
    assert txt.text == "hello"
    assert txt.number == pytest.approx(4.42313353)
    _ = transform(txt, seeds={"frame": 43, "batch": 143, "group": 1143})
    assert txt.text == "hello"
    assert txt.number == pytest.approx(1.79562805)
    _ = transform(txt, seeds={"frame": 44, "batch": 144, "group": 1144})
    assert txt.text == "hello"
    assert txt.number == pytest.approx(1.93032612)


def test_random_transform_class_factory_seeds_setting():
    txt = MockTextTransformable("hello")
    RandomWrapTransform = random_transform_class_factory("RandomWrapTransform", "wrap")
    transform = RandomWrapTransform(
        prob=0.2, param_randomization_rules={"wrapper": {"type": "enum", "choices": ["[]", "()", "<>"]}}, scope="frame"
    )
    assert txt.text == "hello"
    outer_seed = 100
    for _ in range(10):
        random.seed(outer_seed)
        _ = transform(txt, seeds={"frame": 42})
        outer_seed += 1
    assert txt.text == "<<<hello>>>"


def test_random_transform_class_factory_validate_rules():
    RandomWrapTransform = random_transform_class_factory("RandomWrapTransform", "wrap")
    with pytest.raises(AssertionError):
        _ = RandomWrapTransform(param_randomization_rules={"wrapper": {"type": "enum", "range": ["[]", "()", "<>"]}})
    with pytest.raises(AssertionError):
        _ = RandomWrapTransform(param_randomization_rules={"heihei": {"type": "decimal", "range": [-1, 1]}})
    with pytest.raises(AssertionError):
        _ = RandomWrapTransform(param_randomization_rules={"heihei": {"type": "int", "choices": [-1, 1]}})


@pytest.fixture
def cam_im():
    return CameraImage(
        "front",
        "FisheyeCamera",
        np.ones((4, 6, 3)),
        np.ones((4, 6)),
        intrinsic=[10, 20, 0.5, 0.8],
        extrinsic=[np.eye(3), np.zeros((1, 3))],
    )

def test_random_set_intrinsic_param(cam_im):
    transform = RandomSetIntrinsicParam(prob=1.0, jitter_ratio=0.05, scope="group")
    assert transform.jitter_ratio == 0.05
    transform(cam_im, seeds={"frame": 42, "group": 1142})
    assert cam_im.intrinsic == pytest.approx([9.566968, 19.425739, 0.49007237, 0.7770841])


def test_random_set_extrinsic_param(cam_im):
    transform = RandomSetExtrinsicParam(prob=1.0, angle=10, translation=4, scope="batch")
    assert transform.angle == 10 and transform.translation == 4
    transform(cam_im, seeds={"frame": 42, "batch": 142, "group": 1142})
    np.testing.assert_almost_equal(cam_im.extrinsic[0], 
        np.array([[ 0.99801237,  0.00632895,  0.06269974],
       [-0.00451007,  0.99956608, -0.02910845],
       [-0.06285676,  0.02876781,  0.99760786]])
    ) 
    np.testing.assert_almost_equal(cam_im.extrinsic[1], np.array([[-0.111729545, -1.2373623, 2.66947292]]))

