import pytest
import numpy as np

from prefusion.dataset.transform import random_transform_class_factory


class MockTransformable:
    def __init__(self, data: str): self.data = data
    def wrap(self, *, wrapper='()', **kwargs):
        self.data = wrapper[0] + self.data  + wrapper[-1]
        return self

def test_random_transform_class_factory():
    trb = MockTransformable("hello")
    RandomWrapTransform = random_transform_class_factory("RandomWrapTransform", "wrap")
    # transform = RandomWrapTransform(prob=1.0, param_randomization_rules={"wrap": })


