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
    RandomChooseKTransform, RandomBrightness, RandomSharpness, RandomImEqualize
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
        name="camera_image_fisheye",
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
        name="camera_image_perspective",
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


@pytest.fixture()
def camera_imageset():
    return CameraImageSet("camera_image_set", transformables={
        'front_fish': CameraImage(
            name="camera_image_front_fish",
            cam_id='VCAMERA_FISHEYE_FRONT',
            cam_type='FisheyeCamera', 
            img=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye.jpg'), 
            ego_mask=mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_ego_mask.png', flag='grayscale'), 
            extrinsic=(np.array([[ 1.38777878e-16, -5.00000000e-01,  8.66025404e-01],
                                [-1.00000000e+00, -8.32667268e-17,  1.66533454e-16],
                                [ 0.00000000e+00, -8.66025404e-01, -5.00000000e-01]]),
                    np.array([3.81686209, 0.0055232 , 0.74317797])),
            intrinsic=[47.5, 31.5, 24, 24, 0.1, 0, 0, 0]
        ), 
        'front': CameraImage(
            name="camera_image_front",
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
    })



def test_render_intrinsic(fisheye_image, perspective_image, camera_imageset):
    transform = RenderIntrinsic(
        resolutions={
            'VCAMERA_FISHEYE_FRONT': (96, 48),
            'VCAMERA_PERSPECTIVE_FRONT': (96, 48),
        }
    )
    transform(fisheye_image, perspective_image, camera_imageset)
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_intrinsic_result.png')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_intrinsic_result.png')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img)
    np.testing.assert_almost_equal(fisheye_image.intrinsic, [47.5, 23.5, 24.0, 24.0, 0.1, 0, 0, 0])
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img)
    np.testing.assert_almost_equal(perspective_image.intrinsic, [47.5, 23.5, 48.0, 48.0])
    np.testing.assert_almost_equal(camera_imageset.transformables['front'].img, answer_perspective_img)


def test_render_extrinsic(fisheye_image, perspective_image, camera_imageset):
    transform = RenderExtrinsic(
        del_rotations={
            'VCAMERA_FISHEYE_FRONT': [0, -30, 0],
            'VCAMERA_PERSPECTIVE_FRONT': [0, 0, 30]
        }
    )
    transform(fisheye_image, perspective_image, camera_imageset)
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_extrinsic_result.png')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_extrinsic_result.png')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img)
    np.testing.assert_almost_equal(
        fisheye_image.extrinsic[0], 
        np.array([[ 1.20185168e-16,  1.07780618e-10,  1.00000000e+00],
                  [-1.00000000e+00, -8.32667268e-17,  1.66533454e-16],
                  [ 6.93889390e-17, -1.00000000e+00,  1.07780618e-10]])
    )
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img)
    np.testing.assert_almost_equal(
        perspective_image.extrinsic[0], 
        np.array([[ 5.00000000e-01,  1.92296269e-16,  8.66025404e-01],
                  [-8.66025404e-01,  1.11022302e-16,  5.00000000e-01],
                  [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]])
    )
    np.testing.assert_almost_equal(camera_imageset.transformables['front'].img, answer_perspective_img)



def test_random_render_extrinsic(fisheye_image, perspective_image, camera_imageset):
    transform = RandomRenderExtrinsic(
        prob=1,
        angles=[10, 10, 10],
        scope='frame'
    )
    transform(fisheye_image, perspective_image, camera_imageset, seeds={'frame': 520})
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_random_extrinsic_result.png')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_random_extrinsic_result.png')
    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img)
    np.testing.assert_almost_equal(camera_imageset.transformables['front_fish'].img, answer_fisheye_img)
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img)
    np.testing.assert_almost_equal(camera_imageset.transformables['front'].img, answer_perspective_img)


def test_random_render_extrinsic_transformable_scope(fisheye_image, perspective_image, camera_imageset):
    transform = RandomRenderExtrinsic(
        prob=1,
        angles=[10, 10, 10],
        scope='transformable'
    )
    transform(fisheye_image, perspective_image, camera_imageset, seeds={'frame': 520})
    answer_fisheye_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_fisheye_render_random_extrinsic_result_transformable_scope.png')
    answer_perspective_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_perspective_render_random_extrinsic_result_transformable_scope.png')
    answer_camset_front_fish_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_camset_front_fish_render_random_extrinsic_result_transformable_scope.png')
    answer_camset_front_img = mmcv.imread('tests/prefusion/transform/test_render_imgs/test_camset_front_render_random_extrinsic_result_transformable_scope.png')

    np.testing.assert_almost_equal(fisheye_image.img, answer_fisheye_img)
    np.testing.assert_almost_equal(perspective_image.img, answer_perspective_img)
    np.testing.assert_almost_equal(camera_imageset.transformables['front_fish'].img, answer_camset_front_fish_img)
    np.testing.assert_almost_equal(camera_imageset.transformables['front'].img, answer_camset_front_img)


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
        prob=0.7, param_randomization_rules={"wrapper": {"type": "enum", "choices": ["[]", "()", "<>"]}}, scope="frame"
    )
    assert txt.text == "hello"
    for _ in range(3):
        _ = transform(txt, seeds={"frame": 42})
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
        "camera_image_front",
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
    assert cam_im.intrinsic == pytest.approx([9.566968, 19.13393662, 0.47834841, 0.76535746])


def test_random_set_intrinsic_param_transformable_scope(cam_im):
    transform = RandomSetIntrinsicParam(prob=1.0, jitter_ratio=0.05, scope="transformable")
    assert transform.jitter_ratio == 0.05
    transform(cam_im, seeds={"frame": 42, "group": 1142})
    assert cam_im.intrinsic == pytest.approx([9.53855184, 19.07710368, 0.47692759, 0.763084147])


def test_random_set_extrinsic_param(cam_im):
    transform = RandomSetExtrinsicParam(prob=1.0, angle=10, translation=4, scope="batch")
    assert transform.angle == 10 and transform.translation == 4
    transform(cam_im, seeds={"frame": 42, "batch": 142, "group": 1142})
    np.testing.assert_almost_equal(cam_im.extrinsic[0], 
        np.array([[ 0.99801237,  0.00632895,  0.06269974],
                  [-0.00451007,  0.99956608, -0.02910845],
                  [-0.06285676,  0.02876781,  0.99760786]])
    ) 
    np.testing.assert_almost_equal(cam_im.extrinsic[1], np.array([[-0.11172955, -1.2373623, 2.66947292]]))


class MockNumberTransformable:
    def __init__(self, number: float):
        self.number = number

    def add(self, *, a=0, **kwargs): 
        self.number += a
        return self
    
    def multiply(self, *, f=0, **kwargs):
        self.number *= f
        return self
    
    def abs(self, **kwargs):
        self.number = abs(self.number)
        return self

class MockRandomAddTransform:
    def __init__(self, prob=0.5, scope="frame"): 
        self.prob = prob
        self.scope = scope
    def __call__(self, *transformables, seeds=None):
        random.seed(seeds[self.scope])
        for t in transformables:
            t.add(a=random.randint(-10, 10))

class MockAbsTransform:
    def __init__(self, scope='frame'): self.scope = scope
    def __call__(self, *transformables, **kwargs):
        for t in transformables:
            t.abs()

class MockRandomMultiplyTransform:
    def __init__(self, prob=0.5, scope="frame"): 
        self.prob = prob
        self.scope = scope
    def __call__(self, *transformables, seeds=None):
        random.seed(seeds[self.scope])
        for t in transformables:
            t.multiply(f=random.randint(-5, 5))

def test_random_choose_one():
    random_choose_k = RandomChooseKTransform(
        transforms=[
            MockRandomAddTransform(prob=1., scope='frame'),
            MockRandomMultiplyTransform(prob=1., scope='group'),
            MockAbsTransform(),
        ],
        prob=1.0,
        K=1,
    )
    x = MockNumberTransformable(-2)
    np.random.seed(42)
    random_choose_k(x, seeds={'frame': 2, 'batch': 4, 'group': 8})
    assert x.number == -11 # add(-2, -9)


def test_random_choose_k_basic():
    random_choose_k = RandomChooseKTransform(
        transforms=[
            MockRandomAddTransform(prob=1., scope='frame'),
            MockRandomMultiplyTransform(prob=1., scope='batch'),
            MockAbsTransform(),
        ],
        prob=1.0,
        K=2,
    )
    x = MockNumberTransformable(-2)
    np.random.seed(42)
    random_choose_k(x, seeds={'frame': 2, 'batch': 4, 'group': 8})
    assert x.number == 22  # multiply(add(-2, -9), -2) = 22 


def test_random_isp_delegate_transform():
    delegate_transform = RandomChooseKTransform(
        transforms=[
            RandomBrightness(prob=0.6, param_randomization_rules={"brightness": {"type": "float", "range": [0.5, 2.0]}}, scope='batch'),
            RandomSharpness(prob=0.6, param_randomization_rules={"sharpness": {"type": "float", "range": [0.0, 2.0]}}, scope='batch'),
            RandomImEqualize(prob=0.1),
        ],
        prob=1.0,
        K=3,
    )
    np.random.seed(85)  # for np.random.choice(transforms): RandomSharpness -> RandomBrightness -> RandomImEqualize
    random.seed(77)  # for random.random() <= probs: [0.32590, 0.240493, 0.82255]
    im = CameraImage("camera_image_front", "front", "FisheyeCamera", np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3), np.ones((4, 6, 3)), (np.eye(3), np.eye(3)), np.eye(3))
    delegate_transform(im, seeds={'frame': 2, 'batch': 4, 'group': 8})
    np.testing.assert_almost_equal(im.img, np.array(
        [[[ 4,  5,  8, 11, 13, 15],
        [15, 17, 20, 23, 25, 27],
        [30, 33, 35, 38, 40, 42],
        [42, 44, 46, 49, 52, 53]],

       [[ 5,  6,  9, 11, 14, 16],
        [16, 18, 21, 23, 26, 28],
        [31, 34, 36, 39, 41, 43],
        [43, 45, 47, 50, 52, 54]],

       [[ 5,  7, 10, 12, 15, 17],
        [17, 19, 22, 24, 27, 29],
        [32, 35, 37, 40, 42, 44],
        [44, 46, 48, 51, 53, 55]]]
    ).transpose(1, 2, 0))
