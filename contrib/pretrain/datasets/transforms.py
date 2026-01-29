from prefusion.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import LoadAnnotations as SEGLoadAnnotations
from mmdet.datasets.transforms import LoadAnnotations as DETLoadAnnotations
import numpy as np
import mmengine.fileio as fileio
import mmcv
from mmcv import Resize as MMCV_Resize
from mmcv import RandomResize as MMCV_RandomResize
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from PIL import Image as PILImage   
import warnings
from mmcv.image.geometric import _scale_size
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmcv.transforms import LoadImageFromFile as LoadImageFromFile_mmcv



__all__ = ['LoadAnnotationsPretrain', "RandomResize", "Resize", "PackSegInputs", "DetLoadAnnotations",
           "FusionLoadImageFromFile"
           ]

@TRANSFORMS.register_module()
class LoadAnnotationsPretrain(SEGLoadAnnotations):
    def __init__(self, with_depth, with_seg_mask, **kwargs):
        super().__init__(**kwargs)
        
        self.with_depth = with_depth
        self.with_seg_mask = with_seg_mask

    def _load_depth(self, results: dict) -> None:
        if results.get('depth_path', None) is not None:
            results['gt_depth'] = np.load(results['depth_path'])['depth'].astype(np.float32)
        else:
            results['gt_depth'] = (np.zeros(results['img'].shape[:2])-1).astype(np.float32)
        results['seg_fields'].append('gt_depth')    
        return results

    def _load_seg_mask(self, results: dict) -> None:
        if results.get('seg_mask_path', None) is not None:
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
        else:
            results['gt_seg_mask'] = np.zeros(results['img'].shape[:2])
        results['seg_fields'].append('gt_seg_mask')
        return results  
    
    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if results["scene_name"] != "avp":
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)
            semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)[..., 0]
            semantic_seg[semantic_seg==255] = 27
            semantic_seg[semantic_seg==0] = 27
            semantic_seg -= 1
            num_classes = 27
            rows, cols = semantic_seg.shape
            gt_semantic_seg = np.zeros((rows, cols, num_classes))
            gt_semantic_seg[np.arange(rows)[:, None], np.arange(cols), semantic_seg] = 1
            
        else:
            gt_semantic_seg = np.array(PILImage.open(results['seg_map_path'])).astype(np.float32)
            segmap = gt_semantic_seg.transpose(2,0,1)
            gt_semantic_seg = segmap.reshape(segmap.shape[0]*9, segmap.shape[1], int(segmap.shape[2]/9)).transpose(1,2,0)
            # ignore = gt_semantic_seg[..., -1]
            # gt_semantic_seg = gt_semantic_seg[..., :-1]
            # for i in range(gt_semantic_seg.shape[-1]):
            #     gt_semantic_seg[..., i][ignore==1] = 255
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
class Resize(MMCV_Resize):
    """Resize images & bbox & seg.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor']
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

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


    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        self._resize_depth(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
    


@TRANSFORMS.register_module()
class RandomResize(MMCV_RandomResize):
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
                # warnings.warn('Please pay attention your ground truth '
                #               'segmentation map, usually the segmentation '
                #               'map is 2D, but got '
                #               f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64).transpose(2,0,1))
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
    
@TRANSFORMS.register_module()
class DetLoadAnnotations(DETLoadAnnotations):
    def __init__(self, with_bbox=True, with_label=True, with_seg=False, with_depth=False, with_seg_mask=False, **kwargs):
        super().__init__(with_bbox=with_bbox, with_label=with_label, with_seg=with_seg, **kwargs)
        self.with_depth = with_depth
        self.with_seg_mask = with_seg_mask


@TRANSFORMS.register_module()
class FusionLoadImageFromFile(LoadImageFromFile_mmcv):
    """Load image from file.

    Args:
        to_float32 (bool): Whether to convert the loaded image to float32.
            Defaults to False.
        color_type (str): Color type of loaded image. Defaults to 'color'.
        imdecode_backend (str): Backend used for imdecode. Defaults to 'pillow'.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if img is None:
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results