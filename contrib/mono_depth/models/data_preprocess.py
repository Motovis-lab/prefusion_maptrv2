from prefusion.registry import MODELS
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
from pathlib import Path as P
import torch
import numpy as np
from torch import Tensor
from mmdet3d.structures.det3d_data_sample import SampleList
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DetDataPreprocessor
from mmdet3d.utils import OptConfigType
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
import matplotlib.cm as cm
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt 
from contrib.fastbev_det.utils.utils import get_cam_corners, intrinsics_matrix, get_3d_lines, get_bev_lines, get_corners_with_angles, get_bev_lines_cylinder
import cv2

__all__ = ["MonoDepthDataPreprocess"]
# B N C H W 
@MODELS.register_module()
class MonoDepthDataPreprocess(DetDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be divisible by
            ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic segmentation
            maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 IMG_KEYS = [],
                 label_type = [],
                 batch_size = 1,
                 group_size = 1,
                 label_start_idx = 1,
                 ) -> None:
        super(MonoDepthDataPreprocess, self).__init__(
            mean=mean,
            std=std,
            )
        self.IMG_KEYS = IMG_KEYS
        self.label_type = label_type
        self.label_start_idx = label_start_idx
        self.batch_size = batch_size
        self.group_size = group_size

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict or List[dict]): Data from dataloader. The dict contains
                the whole batch data, when it is a list[dict], the list
                indicates test time augmentation.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict or List[dict]: Data in the same format as the model input.
        """
        # data = data[0]
        # for debug dataset output data
        # for i in range(len(data)):
        #     self.show_dataset_output(data[i])
        fish_data = self.cast_data(self.collect_fish_data(data))
        pv_data = self.cast_data(self.collect_pv_data(data))
        front_data = self.cast_data(self.collect_front_data(data))
        delta_pose = self.cast_data(self.collect_delta_pose(data))
        frame_ids = [x['index_info'].frame_id for x in data]
        frame_exists = dict(
            prev_exists=[],
            next_exists=[]
        )
        for x in data:
            if x['index_info'].as_dict()['prev'] is None:
                frame_exists['prev_exists'].append(None)
            else:
                frame_exists['prev_exists'].append(x['index_info'].as_dict()['prev']['frame_id'])
            
            if x['index_info'].as_dict()['next'] is None:
                frame_exists['next_exists'].append(None)
            else:
                frame_exists['next_exists'].append(x['index_info'].as_dict()['next']['frame_id'])

        return {'batch_data':{'fish_data': fish_data, 'front_data':front_data, 'pv_data':pv_data}, 'frame_timestamp':frame_ids, 'delta_pose': delta_pose, 'frame_exists':frame_exists, 'ori_data': data}


    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        # channel transform
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        _batch_img = _batch_img.float()
        # Normalization.
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean.cpu()) / self.std.cpu()
        return _batch_img

    def collect_fish_data(self, data: list):
        camera_types = [x for x in self.IMG_KEYS if 'FISH' in x]
        fish_data = self.collect_data(data, camera_types)
        
        return fish_data

    def collect_pv_data(self, data: list):
        camera_types = [x for x in self.IMG_KEYS if "PERSPECTIVE" in x and "VCAMERA_PERSPECTIVE_FRONT" != x]
        pv_data = self.collect_data(data, camera_types)

        return pv_data
    
    def collect_front_data(self, data: list):
        camera_types = ["VCAMERA_PERSPECTIVE_FRONT"]
        front_data = self.collect_data(data, camera_types)

        return front_data

    def collect_data(self, data, camera_types: list): 
        batch_num = len(data)
        result = dict()
        imgs = []
        depth = []
        extrinsic = []
        intrinsic = []
        ego_mask = []
        for num in range(batch_num):
            single_data = data[num]
            for cam_id in camera_types:
                cam_id_transformable = single_data['transformables']["camera_images"].transformables[cam_id]
                tmp_extrinsic = np.eye(4,4)
                tmp_extrinsic[:3, :3] = cam_id_transformable.extrinsic[0]
                tmp_extrinsic[:3, 3] = cam_id_transformable.extrinsic[1]
                extrinsic.append(torch.from_numpy(tmp_extrinsic).to(torch.float32))
                intrinsic.append(torch.from_numpy(np.array(cam_id_transformable.intrinsic)).to(torch.float32))
                imgs.append(self.preprocess_img(torch.from_numpy(cam_id_transformable.img.transpose(2, 0, 1))))
                ego_mask.append(torch.from_numpy(cam_id_transformable.ego_mask))
                depth.append(torch.from_numpy(single_data['transformables']['camera_depths'].transformables[cam_id].img))
                
        result['imgs'] = torch.stack(imgs, dim=0)
        result['depth'] = torch.stack(depth, dim=0)
        result['extrinsic'] = torch.stack(extrinsic, dim=0)
        result['intrinsic'] = torch.stack(intrinsic, dim=0)
        result['ego_mask'] = torch.stack(ego_mask, dim=0)

        return result 
    
    def collect_delta_pose(self, data: list):
        batch_num = len(data)
        result = dict()
        delta = []
        for num in range(batch_num):
            single_data = data[num]
            transformable = single_data['transformables']["ego_poses"]
            cur_pose = transformable.transformables['0']
            if '-1' not in transformable.transformables:
                pre_pose = transformable.transformables['0']
            else:
                pre_pose = transformable.transformables['-1']
            delta_rotation = pre_pose.rotation.T @ cur_pose.rotation
            delta_translation = pre_pose.rotation.T @ (cur_pose.translation - pre_pose.translation)
            delta_T = torch.eye(4)
            delta_T[:3, :3] = torch.tensor(delta_rotation)
            delta_T[:3, 3:] = torch.tensor(delta_translation)
            delta.append(delta_T)
        return torch.stack(delta, dim=0)

    def show_dataset_output(self, data, debug_data=None):
        """
            show the label with process intrinsic and extrinsic and label
        """
        IMG_KEYS = [
            'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK', 
            'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
            ]
        dump_root = P(f"./work_dirs/show_datapro/")
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_file = dump_root / P(str(data['frame_id'])+'.jpg')

        gt_corners = [] 
        gt_cylinders = []
        for label_type in self.label_type[self.label_start_idx: ]:
            label_data = data['transformables'][label_type].data['elements']
            for box_3d in label_data:
                box = np.array(box_3d['translation'].reshape(-1).tolist() + box_3d['size'] + [Quaternion(matrix=box_3d['rotation']).yaw_pitch_roll[0]] + [0, 0])
                if np.linalg.norm(box[:2]) <= 1e5:
                    corners = get_corners_with_angles(box[None], box_3d['rotation'].T)[0]
                    gt_corners.append(corners)
        
        plt.figure(figsize=(24, 8))
        row = 5 

        for i, k in enumerate(self.IMG_KEYS):
            # Draw camera views
            fig_idx = i + 1 if i < row else i + 2
            plt.subplot(2, 6, fig_idx)

            # Set camera attributes
            plt.title(k)
            plt.axis('off')
            img = data['transformables']['camera_images'][k].data['img']
            W, H = img.shape[1], img.shape[0]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.xlim(0, W)
            plt.ylim(H, 0)

            # Draw images
            plt.imshow(img)

            # Draw 3D gt
            for corners in gt_corners:
                cam_corners = get_cam_corners(
                    corners,
                    data['transformables']['camera_images'][k].data["extrinsic"][1],
                    Quaternion(matrix=data['transformables']['camera_images'][k].data["extrinsic"][0], atol=1e-06),
                    intrinsics_matrix(data['transformables']['camera_images'][k].data['intrinsic'][:4]))
                lines = get_3d_lines(cam_corners)
                for line in lines:
                    plt.plot(line[0],
                            line[1],
                            c=cm.get_cmap('tab10')(4)
                            )
            for corners in gt_cylinders:
                cam_corners = get_cam_corners(
                    corners,
                    data['transformables']['camera_images'][k].data["extrinsic"][1],
                    Quaternion(matrix=data['transformables']['camera_images'][k].data["extrinsic"][0], atol=1e-06),
                    intrinsics_matrix(data['transformables']['camera_images'][k].data['intrinsic'][:4]))
                
                bottom_cam_corner = cam_corners[:100, :]
                up_cam_corner = cam_corners[100:, :]

                plt.plot(bottom_cam_corner[:, 0],
                        bottom_cam_corner[:, 1],
                        c=cm.get_cmap('tab10')(4)
                        )
                plt.plot(up_cam_corner[:, 0],
                        up_cam_corner[:, 1],
                        c=cm.get_cmap('tab10')(4)
                        )
                for i in range(0, 100, 10):
                    plt.plot([bottom_cam_corner[i, 0], up_cam_corner[i, 0]], [bottom_cam_corner[i, 1], up_cam_corner[i, 1]], 'g-')
            # for box in info['box_2d'][k]['box']:
            #     x1, y1, x2, y2 = box
            #     w = x2 - x1
            #     h = y2 - y1
            #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
            #     plt.gca().add_patch(rect)
        
        # Draw BEV
        plt.subplot(1, 6, 6)

        # Set BEV attributes
        plt.title('LIDAR_TOP')
        plt.axis('equal')
        plt.xlim(-15, 15)
        plt.ylim(-15, 40)

        # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
        # Draw BEV GT boxes
        for corners in gt_corners:
            lines = get_bev_lines(corners)
            for line in lines:
                plt.plot([-x for x in line[1]],
                        line[0],
                        c='r',
                        label='ground truth')
        for corners in gt_cylinders:
            lines = get_bev_lines_cylinder(corners)
            for line in lines:
                plt.plot([-x for x in line[1]],
                        line[0],
                        c='r',
                        label='ground truth')
        # Set legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(),
                by_label.keys(),
                loc='upper right',
                framealpha=1)

        # Save figure
        plt.tight_layout(w_pad=0, h_pad=2)
        plt.savefig(dump_file)
        # plt.show()