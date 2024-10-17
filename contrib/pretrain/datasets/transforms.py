from prefusion.registry import TRANSFORMS
from mmcv.transforms import LoadAnnotations
import numpy as np
import mmengine.fileio as fileio
import mmcv

__all__ = ['LoadAnnotationsPretrain']

@TRANSFORMS.register_module()
class LoadAnnotationsPretrain(LoadAnnotations):
    def __init__(self, with_depth, with_seg_mask, **kwargs):
        super().__init__(**kwargs)
        
        self.with_depth = with_depth
        self.with_seg_mask = with_seg_mask

    def _load_depth(self, results: dict) -> None:
        results['gt_depth'] = np.load(results['depth_path'])['depth'].astype(np.float32)
        return results

    def _load_seg_mask(self, results: dict) -> None:
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                self.file_client_args, results['seg_mask_path'])
            img_bytes = file_client.get(results['seg_mask_path'])
        else:
            img_bytes = fileio.get(
                results['seg_mask_path'], backend_args=self.backend_args)

        results['gt_seg_mask'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()
        return results  
    
    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        if self.with_depth:
            self._load_depth(results)
        if self.with_seg_mask:
            self._load_seg_mask(results)
        
        return results