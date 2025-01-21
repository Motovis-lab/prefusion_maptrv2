import json
from pathlib import Path
from typing import Optional, Sequence, Union, List, Dict, TYPE_CHECKING
from functools import partial
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch import nn
from mmengine.hooks.hook import Hook
from mmengine.logging import print_log
from scipy.io import savemat
from scipy.spatial.transform import Rotation as R
from copious.data_structure.dict import defaultdict2dict

from prefusion.registry import HOOKS
from .model_utils import save_pred_outputs

if TYPE_CHECKING:
    from prefusion.dataset.transform import EgoPose

DATA_BATCH = Optional[Union[dict, tuple, list]]

__all__ = ["DumpDetectionAsNuscenesJsonHook"]


DEFAULT_ATTR = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}

OBJ_RANGE_THRESH = {
    'car': 50, 
    'truck': 50, 
    'bus': 50, 
    'trailer': 50, 
    'construction_vehicle': 50, 
    'pedestrian': 40, 
    'motorcycle': 40, 
    'bicycle': 40, 
    'traffic_cone': 30, 
    'barrier': 30
}


@HOOKS.register_module()
class DumpDetectionAsNuscenesJsonHook(Hook):
    def __init__(
        self,
        voxel_shape,
        voxel_range,
        det_anno_transformable_keys: List[str],
        pre_conf_thresh: float = 0.3,
        nms_ratio: float = 1.0,
        area_score_thresh: float = 0.5,
    ):
        super().__init__()
        self.voxel_shape = voxel_shape
        self.voxel_range = voxel_range
        self.pre_conf_thresh = pre_conf_thresh
        self.nms_ratio = nms_ratio
        self.area_score_thresh = area_score_thresh
        self.transformable_keys = det_anno_transformable_keys

        def _create_partial_func(func):
            return partial(
                func,
                voxel_shape=self.voxel_shape,
                voxel_range=self.voxel_range,
                pre_conf_thresh=pre_conf_thresh,
                nms_ratio=nms_ratio,
                area_score_thresh=area_score_thresh,
            )

        self.reverse_funcs = {k: _create_partial_func(eval(f"get_{k}")) for k in self.transformable_keys}
        self.results = defaultdict(list)

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[dict, Sequence]] = None,
        mode: str = "test",
    ) -> None:
        for i, (token, dics, ego_poses) in enumerate(zip(data_batch["sample_token"], data_batch["dictionaries"], data_batch["ego_poses"])):
            token = token.value
            ego_pose = ego_poses.transformables["0"]
            for t_key in self.transformable_keys:
                reversed_boxes = get_reversed_boxes(i, token, t_key, outputs, dics[t_key], self.reverse_funcs[t_key])
                nusc_fmt_boxes = format_boxes_as_nuscenes_format(reversed_boxes, ego_pose)
                self.results[token].extend(nusc_fmt_boxes)

    def after_test_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        json_save_path = Path(runner.cfg["work_dir"]) / "nuscenes_detection_results.json"
        with open(json_save_path, "w") as f:
            json.dump(
                {
                    "meta": {
                        "use_camera": True,
                        "use_lidar": False,
                        "use_radar": False,
                        "use_map": False,
                        "use_external": False,
                    },
                    "results": defaultdict2dict(self.results),
                },
                f,
            )
        print_log(f"Detection results has been saved to {json_save_path}")


def get_reversed_boxes(idx, sample_token, transformable_key, pred_planar_dict, dictionary, reverse_func):
    reversed_boxes = reverse_func(
        {
            "cen": pred_planar_dict[transformable_key]["cen"][idx].cpu().float().sigmoid(),
            "seg": pred_planar_dict[transformable_key]["seg"][idx].cpu().float().sigmoid(),
            "reg": pred_planar_dict[transformable_key]["reg"][idx].cpu().float(),
        },
        dictionary,
    )
    _ = [bx.update(sample_token=sample_token) for bx in reversed_boxes]
    return reversed_boxes


def format_boxes_as_nuscenes_format(boxes: List[Dict], ego_pose: "EgoPose"):
    formatted_boxes = []
    for bx in boxes:
        dist_to_ego_origin = np.linalg.norm(bx['size'][:2], 2)
        if dist_to_ego_origin > OBJ_RANGE_THRESH[bx['detection_name']]:
            continue

        bx['attribute_name'] = get_box_attr(bx)
        bx['size'] = [bx['size'][1], bx['size'][0], bx['size'][2]] # size in nusc is [width, length, height]

        # convert box location to world coord sys
        bx_pos_rot_ego = mat4x4(bx['translation'], bx['rotation'])
        ego_to_world = mat4x4(ego_pose.translation, ego_pose.rotation)
        bx_pos_rot_world = ego_to_world @ bx_pos_rot_ego
        bx['translation'] = bx_pos_rot_world[:3, 3].flatten().tolist()
        bx['rotation'] = R.from_matrix(bx_pos_rot_world[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist() # nusc uses pyquaternion repr
        formatted_boxes.append(bx)
    return formatted_boxes


def get_box_attr(bx):
    if np.sqrt(bx['velocity'][0]**2 + bx['velocity'][1]**2) > 0.2:
        if bx['detection_name'] in [
                'car',
                'construction_vehicle',
                'bus',
                'truck',
                'trailer',
        ]:
            attr = 'vehicle.moving'
        elif bx['detection_name'] in ['bicycle', 'motorcycle']:
            attr = 'cycle.with_rider'
        else:
            attr = DEFAULT_ATTR[bx['detection_name']]
    else:
        if bx['detection_name'] in ['pedestrian']:
            attr = 'pedestrian.standing'
        elif bx['detection_name'] in ['bus']:
            attr = 'vehicle.stopped'
        else:
            attr = DEFAULT_ATTR[bx['detection_name']]
    return attr


def mat4x4(translation, rotation):
    _mat = np.eye(4)
    _mat[:3, :3] = rotation
    _mat[:3, 3] = np.array(translation).flatten()
    return _mat


def remove_boxes_by_area_score_(pred, area_score_thresh=0.5):
    for i in range(len(pred) - 1, -1, -1):
        if pred[i]["area_score"] < area_score_thresh:
            del pred[i]


def get_bbox_3d(
    tensor_dict,
    dictionary,
    voxel_shape=None,
    voxel_range=None,
    pre_conf_thresh=0.3,
    nms_ratio=1.0,
    area_score_thresh=0.5,
):
    from prefusion.dataset.tensor_smith import PlanarBbox3D

    pbox = PlanarBbox3D(
        voxel_shape=voxel_shape,
        voxel_range=voxel_range,
        reverse_pre_conf=pre_conf_thresh,
        reverse_nms_ratio=nms_ratio,
    )
    reversed_pbox = pbox.reverse(tensor_dict)
    remove_boxes_by_area_score_(reversed_pbox, area_score_thresh=area_score_thresh)
    return [
        {
            "translation": bx["translation"].tolist(),
            "size": bx["size"].tolist(),
            "rotation": bx["rotation"],
            "velocity": bx["velocity"][:2].tolist(),
            "detection_name": dictionary["classes"][bx["confs"][1:].argmax()],
            "detection_score": float(bx["confs"][1:].max()),
            "attribute_name": "",
        }
        for bx in reversed_pbox
    ]


def get_bbox_3d_rect_cuboid(
    tensor_dict,
    dictionary,
    voxel_shape=None,
    voxel_range=None,
    pre_conf_thresh=0.3,
    nms_ratio=1.0,
    area_score_thresh=0.5,
):
    from prefusion.dataset.tensor_smith import PlanarRectangularCuboid

    pbox = PlanarRectangularCuboid(
        voxel_shape=voxel_shape, voxel_range=voxel_range, reverse_pre_conf=pre_conf_thresh, reverse_nms_ratio=nms_ratio
    )
    reversed_pbox = pbox.reverse(tensor_dict)
    remove_boxes_by_area_score_(reversed_pbox, area_score_thresh=area_score_thresh)
    return [
        {
            "translation": bx["translation"].tolist(),
            "size": bx["size"].tolist(),
            "rotation": bx["rotation"],
            "velocity": [0, 0],
            "detection_name": dictionary["classes"][bx["confs"][1:].argmax()],
            "detection_score": float(bx["confs"][1:].max()),
            "attribute_name": "",
        }
        for bx in reversed_pbox
    ]


def get_bbox_3d_cylinder(
    tensor_dict,
    dictionary,
    voxel_shape=None,
    voxel_range=None,
    pre_conf_thresh=0.3,
    nms_ratio=1.0,
    area_score_thresh=0.5,
):
    from prefusion.dataset.tensor_smith import PlanarCylinder3D

    pbox = PlanarCylinder3D(
        voxel_shape=voxel_shape, voxel_range=voxel_range, reverse_pre_conf=pre_conf_thresh, reverse_nms_ratio=nms_ratio
    )
    reversed_pbox = pbox.reverse(tensor_dict)
    remove_boxes_by_area_score_(reversed_pbox, area_score_thresh=area_score_thresh)
    return [
        {
            "translation": bx["translation"].tolist(),
            "size": [float(bx["radius"]) * 2, float(bx["radius"]) * 2, float(bx["height"])],
            "rotation": np.eye(3),
            "velocity": [0, 0],
            "detection_name": dictionary["classes"][bx["confs"][1:].argmax()],
            "detection_score": float(bx["confs"][1:].max()),
            "attribute_name": "",
        }
        for bx in reversed_pbox
    ]


def get_bbox_3d_oriented_cylinder(
    tensor_dict,
    dictionary,
    voxel_shape=None,
    voxel_range=None,
    pre_conf_thresh=0.3,
    nms_ratio=1.0,
    area_score_thresh=0.5,
):
    from prefusion.dataset.tensor_smith import PlanarOrientedCylinder3D

    pbox = PlanarOrientedCylinder3D(
        voxel_shape=voxel_shape, voxel_range=voxel_range, reverse_pre_conf=pre_conf_thresh, reverse_nms_ratio=nms_ratio
    )
    reversed_pbox = pbox.reverse(tensor_dict)
    remove_boxes_by_area_score_(reversed_pbox, area_score_thresh=area_score_thresh)
    return [
        {
            "translation": bx["translation"].tolist(),
            "size": bx["size"].tolist(),
            "rotation": bx["rotation"],
            "velocity": bx["velocity"][:2].tolist(),
            "detection_name": dictionary["classes"][bx["confs"][1:].argmax()],
            "detection_score": float(bx["confs"][1:].max()),
            "attribute_name": "",
        }
        for bx in reversed_pbox
    ]


@HOOKS.register_module()
class DumpPlanarPredResultsHookAPA(Hook):
    def __init__(
        self,
        tensor_smith_dict,
        dictionary_dict,
        save_dir=None
    ):
        super().__init__()
        self.tensor_smith_dict = tensor_smith_dict
        self.dictionary_dict = dictionary_dict
        self.save_dir = save_dir

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[dict, Sequence]] = None,
        mode: str = "test",
    ) -> None:
        # print(data_batch.keys())
        # print(outputs.keys())
        rtn = save_pred_outputs(data_batch, outputs, self.tensor_smith_dict, self.dictionary_dict, self.save_dir)
        return rtn
        

@HOOKS.register_module()
class DeployAndBebugHookAPA(Hook):
    def __init__(
        self,
        tensor_smith_dict,
        dictionary_dict,
        save_dir=None
    ):
        super().__init__()
        self.tensor_smith_dict = tensor_smith_dict
        self.dictionary_dict = dictionary_dict
        if save_dir is None:
            self.save_dir = Path("work_dirs/deploy_and_debug")
        else:
            self.save_dir = Path('work_dirs') / save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Union[dict, Sequence]] = None,
        mode: str = "test",
    ) -> None:
        save_pred_outputs(data_batch, outputs, self.tensor_smith_dict, self.dictionary_dict, self.save_dir)
        
        batch_input_dict = runner.model.data_preprocessor(data_batch)

        ori_model = runner.model.cuda()
        ori_model.eval()

        ## dump backbone model and io data
        model_backbone = ori_model.backbone
        camera_tensors_dict = batch_input_dict['camera_tensors']
        camera_lookups = batch_input_dict['camera_lookups']
        camera_feats_dict = {}
        camera_feats_dict_cpu = {}
        for cam_id in camera_tensors_dict:
            camera_feats_dict[cam_id] = model_backbone(camera_tensors_dict[cam_id])
            camera_feats_dict_cpu[cam_id] = camera_feats_dict[cam_id].cpu().detach().numpy()
        torch.onnx.export(model_backbone, camera_tensors_dict[cam_id], str(self.save_dir / "phase_backbone_v9.onnx"), verbose=True,
                          input_names=['camera_img'], output_names=['camera_feats'],
                          opset_version=9)
        savemat(str(self.save_dir / 'camera_tensors.mat'), data_batch['camera_tensors'])
        savemat(str(self.save_dir / 'camera_lookups.mat'), data_batch['camera_lookups'][0])
        savemat(str(self.save_dir / 'camera_feats.mat'), camera_feats_dict_cpu)

        ## dump spatial_transform io data
        spatial_transform = ori_model.spatial_transform
        spatial_transform.dump_voxel_feats = True
        _, voxel_feats = spatial_transform(camera_feats_dict, camera_lookups)
        savemat(str(self.save_dir / 'voxel_feats.mat'), {'voxel_feats': voxel_feats.cpu().detach().numpy()})

        ## dump bev model and io data
        model_bev = DumpBevModel(
            channel_reduction=spatial_transform.channel_reduction,
            voxel_encoder=ori_model.voxel_encoder,
            head_bbox_3d=ori_model.head_bbox_3d,
            head_parkingslot_3d=ori_model.head_parkingslot_3d,
            head_occ_sdf_bev=ori_model.head_occ_sdf_bev
        )
        out_bbox_3d_seg, out_bbox_3d_reg, out_parkingslot_3d_seg, out_parkingslot_3d_reg, out_occ_sdf_bev_seg, out_occ_sdf_bev_reg = model_bev(voxel_feats)
        torch.onnx.export(model_bev, voxel_feats, str(self.save_dir / "phase_bev_v9.onnx"), verbose=True,
                          input_names=['voxel_feats'], output_names=[
                            'out_bbox_3d_seg', 'out_bbox_3d_reg', 'out_parkingslot_3d_seg', 'out_parkingslot_3d_reg', 'out_occ_sdf_bev_seg', 'out_occ_sdf_bev_reg'],
                            opset_version=9)
        savemat(str(self.save_dir / 'bev_outputs.mat'), {
            'out_bbox_3d_seg': out_bbox_3d_seg.cpu().detach().numpy(),
            'out_bbox_3d_reg': out_bbox_3d_reg.cpu().detach().numpy(),
            'out_parkingslot_3d_seg': out_parkingslot_3d_seg.cpu().detach().numpy(),
            'out_parkingslot_3d_reg': out_parkingslot_3d_reg.cpu().detach().numpy(),
            'out_occ_sdf_bev_seg': out_occ_sdf_bev_seg.cpu().detach().numpy(),
            'out_occ_sdf_bev_reg': out_occ_sdf_bev_reg.cpu().detach().numpy()
        })
        assert False
        # return rtn


class DumpBevModel(nn.Module):
    def __init__(self, 
                 channel_reduction,
                 voxel_encoder,
                 head_bbox_3d,
                 head_parkingslot_3d,
                 head_occ_sdf_bev):
        super().__init__()
        self.channel_reduction = channel_reduction
        self.voxel_encoder = voxel_encoder
        self.head_bbox_3d = head_bbox_3d
        self.head_parkingslot_3d = head_parkingslot_3d
        self.head_occ_sdf_bev = head_occ_sdf_bev

    def forward(self, voxel_feats):
        bev_feats = self.channel_reduction(voxel_feats)
        bev_feats = self.voxel_encoder(bev_feats)
        out_bbox_3d_seg, out_bbox_3d_reg = self.head_bbox_3d(bev_feats)
        out_parkingslot_3d_seg, out_parkingslot_3d_reg = self.head_parkingslot_3d(bev_feats)
        out_occ_sdf_bev_seg, out_occ_sdf_bev_reg = self.head_occ_sdf_bev(bev_feats)
        return out_bbox_3d_seg, out_bbox_3d_reg, out_parkingslot_3d_seg, out_parkingslot_3d_reg, out_occ_sdf_bev_seg, out_occ_sdf_bev_reg