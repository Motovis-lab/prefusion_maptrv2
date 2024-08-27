import random
from typing import Tuple

import pytest
import numpy as np

from prefusion.dataset.transform import (
    random_transform_class_factory,
    CameraImage,
    RandomSetIntrinsicParam,
    RandomSetExtrinsicParam,
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

