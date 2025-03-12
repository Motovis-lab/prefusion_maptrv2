
from prefusion.registry import MODEL_FEEDERS


__all__ = ["BaseModelFeeder"]


@MODEL_FEEDERS.register_module()
class BaseModelFeeder:
    """BaseModelFeeder.

    An alternative implementation of data_preprocessor.

    Args
    ----
    Any: Any parameter or keyword arguments.

    """

    def __init__(self, *args, **kwargs) -> None:
        pass


    def process(self, frame_batch: list) -> dict | list:
        """
        Process frame_batch, make it ready for model inputs

        Parameters
        ----------
        frame_batch : list
            list of input_dicts

        Returns
        -------
        processed_frame_batch: dict | list
        }
        """
        processed_frame_batch = frame_batch
        return processed_frame_batch

    def __call__(self, group_batch: list):
        processed_group_batch = []
        for frame_batch in group_batch:
            processed_group_batch.append(self.process(frame_batch))
        
        return processed_group_batch

