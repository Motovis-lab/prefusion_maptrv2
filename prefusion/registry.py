from mmengine import Registry

from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS

TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['prefusion.dataset.transform']
)