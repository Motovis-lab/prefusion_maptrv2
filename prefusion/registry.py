from mmengine import Registry
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS


# base data struct that can be transformed
TRANSFORMABLES = Registry(
    'transformable',
    locations=['prefusion.dataset.transform']
)


TRANSFORMS = Registry(
    'transform',
    # parent=MMENGINE_TRANSFORMS,
    locations=['prefusion.dataset.transform']
)


TENSOR_SMITHS = Registry(
    'tensor_smith',
    locations=['prefusion.dataset.tensor_smith']
)


# an alternative way for data preprocessing
MODEL_FEEDERS = Registry(
    'model_feeder',
    locations=['prefusion.dataset.model_feeder']
)