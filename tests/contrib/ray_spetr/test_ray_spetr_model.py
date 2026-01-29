from prefusion.registry import MODELS
import torch
from mmengine.config import Config

def test_vov_ray_spetr_model():
    cfg = Config.fromfile('contrib/ray_spetr/configs/streampetr_nusc_vov39_overfit.py')
    ray_spetr = MODELS.build(cfg.model)
    # print(ray_spetr.img_backbone)
    # print(ray_spetr.img_neck)   
    assert len(list((ray_spetr.img_backbone.named_modules()))) == 183, "Number of modules in the backbone is not correct"
    assert len(list((ray_spetr.img_neck.named_modules()))) == 11, "Number of modules in the neck is not correct"


def test_r50_spetr_model():
    cfg = Config.fromfile('contrib/petr/configs/streampetr_nusc_r50_overfit.py')
    spetr = MODELS.build(cfg.model)
    # print(spetr.img_backbone)
    # print(spetr.img_neck) 
    assert spetr.img_backbone.depth == 50, "Backbone depth is not correct"
    assert len(list((spetr.img_neck.named_modules()))) == 9, "Number of modules in the neck is not correct"