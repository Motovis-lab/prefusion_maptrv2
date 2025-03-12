from typing import Union
from mmengine.model import BaseModel as MMBaseModel
from prefusion.registry import MODELS

__all__ = ['BaseModel']

@MODELS.register_module()
class BaseModel(MMBaseModel):
    
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='tensor')  # type: ignore