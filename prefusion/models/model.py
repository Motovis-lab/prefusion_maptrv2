from typing import Union
from mmengine.model import BaseModel as MMBaseModel


class BaseModel(MMBaseModel):
    
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='tensor')  # type: ignore