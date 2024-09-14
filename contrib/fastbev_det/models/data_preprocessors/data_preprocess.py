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


# B N C H W 
@MODELS.register_module()
class GroupDataPreprocess(DetDataPreprocessor):
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
                 voxel: bool = False,
                 voxel_type: str = 'hard',
                 voxel_layer: OptConfigType = None,
                 batch_first: bool = True,
                 max_voxels: Optional[int] = None,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 IMG_KEYS = [],
                 label_type = [],
                 batch_size = 1,
                 group_size = 1,
                 label_start_idx = 1,
                 predict_elements = [],
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(GroupDataPreprocess, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)
        self.voxel = voxel
        self.voxel_type = voxel_type
        self.batch_first = batch_first
        self.max_voxels = max_voxels
        self.IMG_KEYS = IMG_KEYS
        self.label_type = label_type
        self.label_start_idx = label_start_idx
        self.predict_elements = predict_elements
        self.batch_size = batch_size
        self.group_size = group_size
        if voxel:
            self.voxel_layer = VoxelizationByGridShape(**voxel_layer)

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
        data = data[0]
        # for debug dataset output data
        # for i in range(len(data)):
        #     self.show_dataset_output(data[i])
        fish_data = self.cast_data(self.collect_fish_data(data))
        pv_data = self.cast_data(self.collect_pv_data(data))
        front_data = self.cast_data(self.collect_front_data(data))
        targets = self.cast_data(self.collate_targets(data))
        frame_ids = [x['frame_id'] for x in data]
        frame_exists = dict(
            prev_exists = [x['prev_exists'] for x in data],
            next_exists = [x['next_exists'] for x in data]
        )

        return {'batch_data':{'fish_data': fish_data, 'front_data':front_data, 'pv_data':pv_data}, 'targets':targets, 'frame_ids':frame_ids, 'frame_exists':frame_exists, 'ori_data': data}


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

    def collate_targets(self, data: list):
        labels = {x:{} for x in self.label_type[self.label_start_idx: ]}
        batch_num = len(data)

        for label_branch in labels.keys():
            for label_type in self.predict_elements:
                for i in range(batch_num):
                    single_data = data[i]
                    if label_type in labels[label_branch].keys():
                        labels[label_branch][label_type].append(torch.from_numpy(single_data['transformables'][label_branch].data['tensor'][label_type][0])) 
                    else:
                        labels[label_branch].update({label_type:[]})
                        labels[label_branch][label_type].append(torch.from_numpy(single_data['transformables'][label_branch].data['tensor'][label_type][0])) 
                labels[label_branch][label_type] = torch.stack(labels[label_branch][label_type], dim=0)

        return labels

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
        uu = []
        vv = []
        dd = []
        valid_map = []
        valid_map_sampled = []
        norm_density_map = []
        ego_mask = []
        mono_imgs = []
        mono_extrinsic = []
        mono_intrinsic = []
        for num in range(batch_num):
            single_data = data[num]
            for cam_id in camera_types:
                tmp_extrinsic = np.eye(4,4)
                tmp_extrinsic[:3, :3] = single_data['transformables']['camera_images'][cam_id].data['extrinsic'][0]
                tmp_extrinsic[:3, 3] = single_data['transformables']['camera_images'][cam_id].data['extrinsic'][1]
                extrinsic.append(torch.from_numpy(tmp_extrinsic).to(torch.float32))
                intrinsic.append(torch.from_numpy(np.array(single_data['transformables']['camera_images'][cam_id].data['intrinsic'])).to(torch.float32))
                imgs.append(self.preprocess_img(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['img'].transpose(2, 0, 1))))
                uu.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['uu']))
                vv.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['vv']))
                dd.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['dd']))
                valid_map.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['valid_map']))
                valid_map_sampled.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['valid_map_sampled']))
                norm_density_map.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['fast_ray_LUT']['norm_density_map']))
                ego_mask.append(torch.from_numpy(single_data['transformables']['camera_images'][cam_id].data['ego_mask']))
                depth.append(torch.from_numpy(single_data['transformables']['camera_depths'][cam_id].data['dep_img']))
                
        result['imgs'] = torch.stack(imgs, dim=0)
        result['depth'] = torch.stack(depth, dim=0)
        result['extrinsic'] = torch.stack(extrinsic, dim=0)
        result['intrinsic'] = torch.stack(intrinsic, dim=0)
        result['uu'] = torch.stack(uu, dim=0)
        result['vv'] = torch.stack(vv, dim=0)
        result['dd'] = torch.stack(dd, dim=0)
        result['valid_map'] = torch.stack(valid_map, dim=0)
        result['valid_map_sampled'] = torch.stack(valid_map_sampled, dim=0)
        result['norm_density_map'] = torch.stack(norm_density_map, dim=0)
        result['ego_mask'] = torch.stack(ego_mask, dim=0)
        for k in range(-1, 2):
            for num in range(batch_num):
                single_data = data[num]
                for cam_id in camera_types:
                    mono_imgs.append(self.preprocess_img(torch.from_numpy(single_data['transformables']['mono_input_data'][f"frame_{k}"][cam_id].data['img'].transpose(2, 0, 1))))
                    tmp_extrinsic = np.eye(4,4)
                    tmp_extrinsic[:3, :3] = single_data['transformables']['mono_input_data'][f"frame_{k}"][cam_id].data['extrinsic'][0]
                    tmp_extrinsic[:3, 3] = single_data['transformables']['mono_input_data'][f"frame_{k}"][cam_id].data['extrinsic'][1]
                    mono_extrinsic.append(torch.from_numpy(tmp_extrinsic).to(torch.float32))
                    mono_intrinsic.append(torch.from_numpy(np.array(single_data['transformables']['mono_input_data'][f"frame_{k}"][cam_id].data['intrinsic'])).to(torch.float32))
        result['mono_imgs'] = torch.stack(mono_imgs, dim=0)
        result['mono_extrinsic'] = torch.stack(mono_extrinsic, dim=0)
        result['mono_intrinsic'] = torch.stack(mono_intrinsic, dim=0)

        return result 

    @torch.no_grad()
    def voxelize(self, points: List[Tensor],
                 data_samples: SampleList) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        if self.voxel_type == 'hard':
            voxels, coors, num_points, voxel_centers = [], [], [], []
            for i, res in enumerate(points):
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                        self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                            self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)

            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxel_centers = torch.cat(voxel_centers, dim=0)

            voxel_dict['num_points'] = num_points
            voxel_dict['voxel_centers'] = voxel_centers
        elif self.voxel_type == 'dynamic':
            coors = []
            # dynamic voxelization only provide a coors mapping
            for i, res in enumerate(points):
                res_coors = self.voxel_layer(res)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                coors.append(res_coors)
            voxels = torch.cat(points, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'cylindrical':
            voxels, coors = [], []
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                rho = torch.sqrt(res[:, 0]**2 + res[:, 1]**2)
                phi = torch.atan2(res[:, 1], res[:, 0])
                polar_res = torch.stack((rho, phi, res[:, 2]), dim=-1)
                min_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[:3])
                max_bound = polar_res.new_tensor(
                    self.voxel_layer.point_cloud_range[3:])
                try:  # only support PyTorch >= 1.9.0
                    polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                  max_bound)
                except TypeError:
                    polar_res_clamp = polar_res.clone()
                    for coor_idx in range(3):
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] >
                            max_bound[coor_idx]] = max_bound[coor_idx]
                        polar_res_clamp[:, coor_idx][
                            polar_res[:, coor_idx] <
                            min_bound[coor_idx]] = min_bound[coor_idx]
                res_coors = torch.floor(
                    (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                        self.voxel_layer.voxel_size)).int()
                self.get_voxel_seg(res_coors, data_sample)
                res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
                res_voxels = torch.cat((polar_res, res[:, :2], res[:, 3:]),
                                       dim=-1)
                voxels.append(res_voxels)
                coors.append(res_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
        elif self.voxel_type == 'minkunet':
            voxels, coors = [], []
            voxel_size = points[0].new_tensor(self.voxel_layer.voxel_size)
            for i, (res, data_sample) in enumerate(zip(points, data_samples)):
                res_coors = torch.round(res[:, :3] / voxel_size).int()
                res_coors -= res_coors.min(0)[0]

                res_coors_numpy = res_coors.cpu().numpy()
                inds, point2voxel_map = self.sparse_quantize(
                    res_coors_numpy, return_index=True, return_inverse=True)
                point2voxel_map = torch.from_numpy(point2voxel_map).cuda()
                if self.training and self.max_voxels is not None:
                    if len(inds) > self.max_voxels:
                        inds = np.random.choice(
                            inds, self.max_voxels, replace=False)
                inds = torch.from_numpy(inds).cuda()
                if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
                    data_sample.gt_pts_seg.voxel_semantic_mask \
                        = data_sample.gt_pts_seg.pts_semantic_mask[inds]
                res_voxel_coors = res_coors[inds]
                res_voxels = res[inds]
                if self.batch_first:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (1, 0), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, 0]
                else:
                    res_voxel_coors = F.pad(
                        res_voxel_coors, (0, 1), mode='constant', value=i)
                    data_sample.batch_idx = res_voxel_coors[:, -1]
                data_sample.point2voxel_map = point2voxel_map.long()
                voxels.append(res_voxels)
                coors.append(res_voxel_coors)
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)

        else:
            raise ValueError(f'Invalid voxelization type {self.voxel_type}')

        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    def ravel_hash(self, x: np.ndarray) -> np.ndarray:
        """Get voxel coordinates hash for np.unique.

        Args:
            x (np.ndarray): The voxel coordinates of points, Nx3.

        Returns:
            np.ndarray: Voxels coordinates hash.
        """
        assert x.ndim == 2, x.shape

        x = x - np.min(x, axis=0)
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords: np.ndarray,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[np.ndarray]:
        """Sparse Quantization for voxel coordinates used in Minkunet.

        Args:
            coords (np.ndarray): The voxel coordinates of points, Nx3.
            return_index (bool): Whether to return the indices of the unique
                coords, shape (M,).
            return_inverse (bool): Whether to return the indices of the
                original coords, shape (N,).

        Returns:
            List[np.ndarray]: Return index and inverse map if return_index and
            return_inverse is True.
        """
        _, indices, inverse_indices = np.unique(
            self.ravel_hash(coords), return_index=True, return_inverse=True)
        coords = coords[indices]

        outputs = []
        if return_index:
            outputs += [indices]
        if return_inverse:
            outputs += [inverse_indices]
        return outputs
    
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