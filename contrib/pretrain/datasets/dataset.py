from prefusion.registry import DATASETS
from mmengine.dataset import BaseDataset
import mmengine
from typing import Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path as P
import copy
from collections.abc import Mapping
from mmengine import Config
import numpy as np
from mmengine.dataset import Compose


__all__ = ['PretrainDataset', 'PretrainDataset_AVP']

@DATASETS.register_module()
class PretrainDataset(BaseDataset):
    METAINFO = {
        'classes': ('freespace', 'curb', 'lane-mark', 'parking-line', 'deceleraion-hump', 'fence', 'warning-tape', 'motor', 'bicycles',
                    'pedestrians', 'warning-pole', 'warning-cone', 'lifting-pole', 'no-parking', 'obstacle', 'wheel', 'plate',
                    'guideline', 'ground-pin', 'ground-lock', 'pillar', 'zebracross', 'water_barrier', 'fire hyrant', 'Non-motor', 'wall'),
        'palette': [
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32], [0, 0, 142], [0, 0, 70], [250, 170, 160], [128, 64, 64], [70, 130, 180], [81, 0, 81], [150, 100, 100]
        ]
    }
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 camera_types: List=None,
                 test_mode=False,
                 reduce_zero_label=True,
                 lazy_init: bool = False,
                 serialize_data: bool = True,
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 metainfo: Union[Mapping, Config, None] = None,
                 indices=None,
                 max_refetch=1000,
                 **kwargs
                ) -> None:
        self.ann_infos = mmengine.load(P(data_root) / P(ann_file))
        self.data_root = data_root  
        self.camera_types = camera_types
        self.scene_names = [x for x in self.ann_infos]
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray
        self._indices = indices
        self.max_refetch = max_refetch
        self.reduce_zero_label = reduce_zero_label
        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()

    def load_data_list(self) -> List[dict]:
        data_list = []
        for scene_name in self.scene_names:
            frame_ids = [x for x in self.ann_infos[scene_name]['frame_info']]
            for frame_id in frame_ids:
                for camera_type in self.camera_types:
                    data_info = dict(
                        img_path=P(self.data_root) / P(self.ann_infos[scene_name]['frame_info'][frame_id]['camera_image'][camera_type]),
                        seg_map_path=P(self.data_root) / P(self.ann_infos[scene_name]['frame_info'][frame_id]['camera_image_seg'][camera_type]),
                        depth_path=P(self.data_root) / P(self.ann_infos[scene_name]['frame_info'][frame_id]['camera_image_depth'][camera_type]),
                        seg_mask_path=P(self.data_root) / P(self.ann_infos[scene_name]['scene_info']['camera_mask'][camera_type]),
                        scene_name=scene_name,
                        camera_type=camera_type,
                        frame_id=frame_id,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                        swap_seg_labels=None)
                    data_list.append(data_info)
        return data_list
    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


@DATASETS.register_module()
class PretrainDataset_AVP(PretrainDataset):
    def load_data_list(self) -> List[dict]:
        data_list = []
        for scene_name in ["avp"]:
            frame_ids = [x for x in self.ann_infos[scene_name]['frame_info']]
            for frame_id in frame_ids:
                for camera_type in self.camera_types:
                    data_info = dict(
                        img_path=P(self.data_root) / P(self.ann_infos[scene_name]['frame_info'][frame_id]['camera_image'][camera_type]),
                        seg_map_path=P(self.data_root) / P(self.ann_infos[scene_name]['frame_info'][frame_id]['camera_image_seg'][camera_type]),
                        depth_path=None,
                        seg_mask_path=None,
                        scene_name=scene_name,
                        camera_type=camera_type,
                        frame_id=frame_id,
                        reduce_zero_label=self.reduce_zero_label,
                        seg_fields=[],
                        swap_seg_labels=None)
                    data_list.append(data_info)
        return data_list