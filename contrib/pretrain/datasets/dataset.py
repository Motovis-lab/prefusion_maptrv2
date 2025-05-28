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
from mmdet.datasets import XMLDataset as MMdetBaseDetDataset
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union
from mmengine.fileio import get, get_local_path, list_from_file



__all__ = ['PretrainDataset', 'PretrainDataset_AVP', "PretrainDataset_FrontData"]

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
    


@DATASETS.register_module()
class PretrainDataset_FrontData(MMdetBaseDetDataset):
    METAINFO = {
        # 'classes':
        # ('car','mpv','mini','van','bus','lorry','truck','special','adult','child','bicycle','motorcycle',
        #  'tricycle','bicyclist','tricyclist','trafficsign','tunnel_entry','vehicle_front','vehicle_back',
        #  'tricycle_front','tricycle_back','wheel','plate','head','mirror','cabin','info_ts','other_ts','cone',
        #  'bollard','direction_guidance','soft_barrier','guardrail','dontcareregion','front_wheel_point',
        #  'back_wheel_point','suv')
        'classes': ("vehicle", "pedestrian", "bicycle"),
        'sub_classes': {
            ('car','mpv','mini','van','bus','lorry','truck','suv'): "vehicle",
            ('adult', 'child'): "pedestrian",
            ('bicycle', 'motorcycle', 'tricycle', 'bicyclist', 'tricyclist'): "bicycle"
        }
    }
    def __init__(self, reduce_zero_label=False, **kwargs):
        super().__init__(img_subdir="", ann_subdir="", **kwargs)
        self.data_root = kwargs.get('data_root')
        self.ann_file = kwargs.get('ann_file')
        self.pipeline = Compose(kwargs.get('pipeline'))
        self.reduce_zero_label = reduce_zero_label
        self.test_mode = kwargs.get('test_mode', False)
        self.data_list: List[dict] = []
        
    
    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        
        self.cat2label = {
            cat: i
            for i, cat in enumerate(self._metainfo['classes'])
        }

        data_list = []
        if self.ann_file.split('/')[0] != "data":
            self._join_prefix()
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.data_root, img_id + '.jpg')
            xml_path = osp.join(self.data_root,
                                img_id + '.xml')

            raw_img_info = {}
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = file_name
            raw_img_info['xml_path'] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list
    
    def _parse_instance_info(self,
                             raw_ann_info: ET,
                             minus_one: bool = True) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (ElementTree): ElementTree object.
            minus_one (bool): Whether to subtract 1 from the coordinates.
                Defaults to True.

        Returns:
            List[dict]: List of instances.
        """
        instances = []
        for obj in raw_ann_info.findall('object'):
            instance = {}
            name = obj.find('name').text
            if name not in self._metainfo['classes']:
                continue
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]

            # VOC needs to subtract 1 from the coordinates
            if minus_one:
                bbox = [x - 1 for x in bbox]

            ignore = False
            if self.bbox_min_size is not None:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.bbox_min_size or h < self.bbox_min_size:
                    ignore = True
            if difficult or ignore:
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            sub_classes = list(self._metainfo['sub_classes'].keys())
            for sub in sub_classes:
                if name in sub:
                    sub_name = self._metainfo['sub_classes'][sub]
                    break
            instance['bbox_label'] = self.cat2label[sub_name]
            instances.append(instance)
        return instances