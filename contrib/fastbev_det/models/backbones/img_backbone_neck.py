from mmengine.model import BaseModule
from prefusion.registry import MODELS


@MODELS.register_module()
class ImgBackboneNeck(BaseModule):
    def __init__(self,
                 img_backbone_conf = dict(),
                 img_neck_conf = dict(),
                 init_cfg=None):
        super().__init__(init_cfg)
    
        self.backbone = MODELS.build(img_backbone_conf)
        self.neck = MODELS.build(img_neck_conf)

    def forward(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x
