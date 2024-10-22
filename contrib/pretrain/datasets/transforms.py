from prefusion.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import LoadAnnotations
import numpy as np
import mmengine.fileio as fileio
import mmcv
from mmcv import Resize, RandomResize
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import warnings


__all__ = ['LoadAnnotationsPretrain', "RandomResize", "Resize", "PackSegInputs"]

@TRANSFORMS.register_module()
class LoadAnnotationsPretrain(LoadAnnotations):
    def __init__(self, with_depth, with_seg_mask, **kwargs):
        super().__init__(**kwargs)
        
        self.with_depth = with_depth
        self.with_seg_mask = with_seg_mask

    def _load_depth(self, results: dict) -> None:
        results['gt_depth'] = np.load(results['depth_path'])['depth'].astype(np.float32)
        results['seg_fields'].append('gt_depth')
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
            backend=self.imdecode_backend).squeeze()[..., 0]
        results['seg_fields'].append('gt_seg_mask')
        return results  
    
    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)[..., 0]

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
    
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


@TRANSFORMS.register_module()
class Resize(Resize):
    
    def _resize_seg_mask(self, results: dict) -> None:
        """Resize semantic segmentation map mask with ``results['scale']``."""
        if results.get('gt_seg_mask', None) is not None:
            if self.keep_ratio:
                gt_seg_mask = mmcv.imrescale(
                    results['gt_seg_mask'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg_mask = mmcv.imresize(
                    results['gt_seg_mask'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_mask'] = gt_seg_mask
    
    def _resize_depth(self, results: dict) -> None:
        """Resize depth with ``results['scale']``."""
        if results.get('gt_depth', None) is not None:
            if self.keep_ratio:
                gt_depth = mmcv.imrescale(
                    results['gt_depth'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_depth = mmcv.imresize(
                    results['gt_depth'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_depth'] = gt_depth

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        super().transform(results)
        self._resize_seg_mask(results)
        self._resize_depth(results)
        
        return results


@TRANSFORMS.register_module()
class RandomResize(RandomResize):
    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        resize_type: str = 'prefusion.Resize',
        **resize_kwargs,
    ) -> None:

        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})    


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))
        
        if 'gt_depth' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth'][None, ...]))
            data_sample.set_data(dict(gt_depth=PixelData(**gt_depth_data)))
            
        if 'gt_seg_mask' in results:
            gt_seg_mask_data = dict(
                data=to_tensor(results['gt_seg_mask'][None, ...]))
            data_sample.set_data(dict(gt_seg_mask=PixelData(**gt_seg_mask_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str