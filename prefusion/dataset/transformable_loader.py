from typing import Callable, Any, TYPE_CHECKING

from prefusion.registry import TRANSFORMABLE_LOADERS
from prefusion.dataset.tensor_smith import TensorSmith


if TYPE_CHECKING:
    from prefusion.dataset import IndexInfo


def _load_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "load" function')


class TransformableLoader:
    load: Callable[..., Any] = _load_unimplemented


class CameraImageSetLoader(TransformableLoader):
    def load(name: str, index_info: "IndexInfo", loader: TransformableLoader, tensor_smith: TensorSmith = None, **kwargs):
        


class EgoPoseSetLoader(TransformableLoader):
    pass