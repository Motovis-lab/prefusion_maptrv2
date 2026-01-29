import torch
import numpy as np

from typing import List, Union, TYPE_CHECKING
from collections import defaultdict
from copious.data_structure.dict import defaultdict2dict
from prefusion.dataset.model_feeder import BaseModelFeeder
from prefusion.registry import MODEL_FEEDERS

from .voxel_lut import VoxelLookUpTableGenerator
from shapely.geometry import Polygon, Point
from prefusion.dataset import (
    CameraImageSet, EgoPoseSet, LidarPoints,
    Bbox3D, Polyline3D, ParkingSlot3D, OccSdfBev
)

from prefusion.dataset.tensor_smith import (
    PlanarBbox3D, PlanarSquarePillar,
    PlanarCylinder3D, PlanarOrientedCylinder3D,
    PlanarPolyline3D, PlanarPolygon3D,
    PlanarParkingSlot3D, PlanarOccSdfBev,
    PlanarBbox3D_pin, PlanarPolyline3D_laneline, PlanarPolyline3D_laneline_mul
)
import cv2

# TODO: occ2d, should merge multiple frames to one


__all__ = ["FastRayPlanarModelFeeder"]


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray.copy())
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def draw_labelmap_gaussian(img, pt, sigma, type='Gaussian', factor=1, radius=3):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)
    thres= np.exp(- (radius ** 2) / (2 * sigma ** 2))
    # Check that any part of the gaussian is in-bounds
    ul = [round(pt[0]) - 2 * sigma, round(pt[1]) - 2 * sigma]
    br = [round(pt[0]) + 2 * sigma + 1, round(pt[1]) + 2 * sigma + 1]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 4 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # g[g<thres]=0
        # g[g>=thres]=1
        g=g*factor 
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # print(g.shape)
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])


    # print(g.max(),g.min())\
    # print(g_x,img_x, ul[0],br[0],img.shape[1])
    try:
        img[int(img_y[0]):int(img_y[1]), int(img_x[0]):int(img_x[1])] = g[int(g_y[0]):int(g_y[1]), int(g_x[0]):int(g_x[1])]
    except:
        print(g_x,img_x, ul[0],br[0],img.shape[1])
        print('pt=',pt)
        # print()
    # img[img_x[0]:img_x[1], img_y[0]:img_y[1]] = g[g_x[0]:g_x[1], g_y[0]:g_y[1]]     #yan, the original crop process use[y,x]
    # cv2.imwrite('./1.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return to_torch(img)

__all__ = ["FastRayPlanarModelFeeder"]

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray.copy())
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def putVecMaps(centerA, centerB, accumulate_vec_map,count, out_size_x=240, out_size_y=320):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    grid_y = out_size_y
    grid_x = out_size_x
    thre = 2   # limb width
    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print(limb_vec_unit)
    # print 'limb unit vector: {}'.format(limb_vec_unit)
    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    # print(xx.shape)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    limb_vec_unit[0:2]=1
    vec_map = np.zeros([out_size_y,out_size_x,2])

    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    # print(vec_map.type())
    accumulate_vec_map=to_numpy(accumulate_vec_map)

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :])

    accumulate_vec_map += (vec_map[:,:,0])

    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1
    # print(accumulate_vec_map.shape)
    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :])
    count[mask == True] = 0
    return to_torch(accumulate_vec_map),count


def sort_corners_clockwise(pts):
    """将任意四边形点排序为顺时针顺序"""
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]

def get_edges_and_lengths(pts):
    """返回4条边和边长"""
    edges = [(pts[i], pts[(i+1)%4]) for i in range(4)]
    lengths = [np.linalg.norm(e[0] - e[1]) for e in edges]
    return edges, lengths

def get_long_and_short_dirs(pts):
    """根据边长识别长边和短边方向（返回单位向量）"""
    _, lengths = get_edges_and_lengths(pts)
    edges = [(pts[i], pts[(i+1)%4]) for i in range(4)]
    dir_vectors = [e[1] - e[0] for e in edges]
    lengths = np.array(lengths)
    long_idx = np.argmax(lengths)
    short_idx = np.argmin(lengths)
    long_dir = dir_vectors[long_idx] / (np.linalg.norm(dir_vectors[long_idx]) + 1e-6)
    short_dir = dir_vectors[short_idx] / (np.linalg.norm(dir_vectors[short_idx]) + 1e-6)
    return long_dir, short_dir

def point_in_polygon(point, polygon):
    return Polygon(polygon).contains(Point(point))

def get_farthest_long_side(pin_pts, slot_pts):
    """找出挡轮杆中离车位最远的长边"""
    # 整理排序 + 获取方向
    pin_pts_sorted = sort_corners_clockwise(pin_pts)
    slot_pts_sorted = sort_corners_clockwise(slot_pts)

    # 获取挡轮杆长边方向
    pin_long_dir, _ = get_long_and_short_dirs(pin_pts_sorted)
    _, slot_short_dir = get_long_and_short_dirs(slot_pts_sorted)

    # 确保挡轮杆和车位方向基本一致
    if np.abs(np.dot(pin_long_dir, slot_short_dir)) < 0.9:
        return None  # 方向不对，跳过

    # 找出pin的两条长边
    edges, lengths = get_edges_and_lengths(pin_pts_sorted)
    long_edges = [edges[i] for i in range(4) if np.abs(lengths[i] - max(lengths)) < 1e-3]

    # 分别计算两条长边到 slot 的平均距离
    dists = []
    for e in long_edges:
        # d = np.mean([np.min(np.linalg.norm(s - p)) for p in e for s in slot_pts_sorted]) # wrong side
        d = np.mean([np.min([np.linalg.norm(s - p) for s in slot_pts_sorted]) for p in e])
        dists.append(d)

    # 取更远的那条边
    farthest_edge = long_edges[np.argmax(dists)]
    return np.array(farthest_edge)

def match_and_filter(slots_points_bev, pin_points_bev):
    output_pins = []

    for pin in pin_points_bev:
        pin_center = np.mean(pin, axis=0)
        matched_slot = None
        for slot in slots_points_bev:
            if point_in_polygon(pin_center, slot):
                matched_slot = slot
                break
        if matched_slot is None:
            continue  # 无匹配

        edge = get_farthest_long_side(pin, matched_slot)
        if edge is not None:
            output_pins.append(edge)

    return output_pins


@MODEL_FEEDERS.register_module()
class FastRayPlanarModelFeeder(BaseModelFeeder):
    # TODO: for sdf_2d, we should mix tensor across frames

    def __init__(self, 
                 voxel_feature_config: dict, 
                 camera_feature_configs: dict,
                 bilinear_interpolation: bool = True,
                 debug_mode: bool = False):
        super().__init__()
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=self.voxel_feature_config,
            camera_feature_configs=self.camera_feature_configs,
            bilinear_interpolation=bilinear_interpolation
        )
        self.debug_mode = debug_mode


    def pin_to_tensor(self, input_dict) -> torch.Tensor:
        """
        this is the to_tensor method for PlanarBbox3D_pin
        """
        slots_points_bev = [] 
        pin_points_bev = []  
        for transformable in input_dict['transformables'].values():
            match transformable:    
                case Bbox3D() | ParkingSlot3D():
                    anno_ts = transformable.tensor_smith
                    match anno_ts:
                        # get the points of the parking slot
                        case PlanarParkingSlot3D():
                            Z, X, Y = anno_ts.voxel_shape
                            cx, cy, fx, fy = anno_ts.bev_intrinsics
                            for element in transformable.elements:
                                slot_points_3d = (np.float32(element['points']) * [[fx, fy, 1]] + [[cx, cy, 0]])
                                if not anno_ts._valid_slot(slot_points_3d):
                                    continue
                                entrance_length = np.linalg.norm(element['points'][1] - element['points'][0])
                                side_length = np.linalg.norm(element['points'][3] - element['points'][0])
                                if entrance_length > side_length:
                                    slot_points_3d = slot_points_3d[[1, 2, 3, 0]]
                                slot_points_bev = slot_points_3d[..., [1, 0]]
                                slots_points_bev.append(slot_points_bev)
                        #get the points of the wheel stopper
                        case PlanarBbox3D_pin():
                            Z, X, Y = anno_ts.voxel_shape
                            cx, cy, fx, fy = anno_ts.bev_intrinsics
                            for element in transformable.elements:
                                if element['class'] in transformable.dictionary['classes']:
                                    # get the position of the box
                                    center = np.float32(element['translation'][:, 0])
                                    if anno_ts.use_bottom_center:
                                        center -= 0.5 * element['size'][2] * element['rotation'][:, 2]
                                    xvec = element['size'][0] * element['rotation'][:, 0]
                                    yvec = element['size'][1] * element['rotation'][:, 1]
                                    corner_points = np.array([
                                        center + 0.5 * xvec - 0.5 * yvec,
                                        center + 0.5 * xvec + 0.5 * yvec,
                                        center - 0.5 * xvec + 0.5 * yvec,
                                        center - 0.5 * xvec - 0.5 * yvec
                                    ], dtype=np.float32)
                                    points_bev = np.array([
                                        corner_points[..., 1] * fy + cy,
                                        corner_points[..., 0] * fx + cx
                                    ], dtype=np.float32).T
                                    pin_points_bev.append(points_bev)            

        pts_im = np.zeros((1, X*2, Y*2), dtype=np.float32)
        count_pin_seg = np.zeros((X*2, Y*2), dtype=np.float32)
        pin_points_bev=match_and_filter(slots_points_bev, pin_points_bev)

        # heatmap based gt
        # radius = 2
        # sigma = 2
        # for pin in pin_points_bev:
        #     # pts_im[0], count_pin_seg = putVecMaps(np.round(pin[0] * 2).astype(int), np.round(pin[1] * 2).astype(int), pts_im[0], count=count_pin_seg, out_size_x=Y*2, out_size_y=X*2)
        #     # cv2.circle(pts_im[0], tuple(np.round(pin[0]+pin[1]).astype(int)), radius, 1, -1)
        #     pts_im[0] = draw_labelmap_gaussian(pts_im[0], tuple(np.round(pin[0]+pin[1]).astype(int)), sigma=sigma,radius=radius, type='Gaussian', factor=1)

        radius = 3
        # sigma = 2
        for pin in pin_points_bev:
            # pts_im[0], count_pin_seg = putVecMaps(np.round(pin[0] * 2).astype(int), np.round(pin[1] * 2).astype(int), pts_im[0], count=count_pin_seg, out_size_x=Y*2, out_size_y=X*2)
            cv2.circle(pts_im[0], tuple(np.round(pin[0]+pin[1]).astype(int)), radius, 1, -1)
            # pts_im[0] = draw_labelmap_gaussian(pts_im[0], tuple(np.round(pin[0]+pin[1]).astype(int)), sigma=sigma,radius=radius, type='Gaussian', factor=1)

        # cv2.imwrite('pin_points_bev.png', pts_im[0] * 255)
        # pdb.set_trace()
        return torch.tensor(pts_im)


    def pin_exists(self, input_dict) -> bool:
        """
        check if the pin exists in the input_dict
        """
        for transformable in input_dict['transformables'].values():
            match transformable:
                case Bbox3D():
                    anno_ts = transformable.tensor_smith
                    match anno_ts:
                        case PlanarBbox3D_pin():
                            if len(transformable.elements) > 0:
                                return True
        return False

    def process(self, frame_batch: list) -> Union[dict, list]:
        """
        Parameters
        ----------
        frame_batch : list
            list of input_dicts

        Returns
        -------
        dict | list
            processed_frame_batch

        Notes
        -----
        ```
        input_dict = {
            'index_info': index_info,
            'transformables': []
        }
        processed_frame_batch = {
            'index_infos': [index_info, index_info, ...],
            'camera_images': {
                'cam_0': (N, 3, H, W),
                ...
            },
            'camera_lookups': [
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
            ],
            'delta_poses': [],
            'annotations': {
                'bbox_3d_0': {
                    'cen': (N, 1, X, Y)
                    'seg': (N, C, X, Y)
                    'reg': (N, C, X, Y)
                },
                ...
            },
            ...
        }
        ```
        """
        processed_frame_batch = {
            'index_infos': [],
            'camera_tensors': defaultdict(list),
            'camera_lookups': [],
            'lidar_points': [],
            'delta_poses': [],
            'annotations': defaultdict(lambda: defaultdict(list))
        }
        if self.debug_mode:
            processed_frame_batch['transformables'] = []
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            # append transformables
            if self.debug_mode:
                processed_frame_batch['transformables'].append(input_dict['transformables'])
            # use parking slot and wheel stopper to generate pin(挡轮杆) tensors
            # if self.pin_exists(input_dict):
            #     pin_tensor=self.pin_to_tensor(input_dict)
            # batching transformables
            for transformable in input_dict['transformables'].values():
                match transformable:
                    case CameraImageSet():
                        camera_lookup = self.voxel_lut_gen.generate(transformable)
                        for cam_id in camera_lookup:
                            for lut_key in camera_lookup[cam_id]:
                                camera_lookup[cam_id][lut_key] = torch.tensor(
                                    camera_lookup[cam_id][lut_key])
                        processed_frame_batch['camera_lookups'].append(camera_lookup)
                        for cam_id in transformable.transformables:
                            camera_tensor = transformable.transformables[cam_id].tensor['img']
                            processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
                    case LidarPoints():
                        raise NotImplementedError
                    case EgoPoseSet():
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
                        processed_frame_batch['delta_poses'].append(delta_T)
                     
                    case Bbox3D() | Polyline3D() | ParkingSlot3D() | OccSdfBev():
                        annotation_tensor = transformable.tensor
                        anno_ts = transformable.tensor_smith
                        match anno_ts:
                            case (PlanarBbox3D() | PlanarSquarePillar()
                                | PlanarCylinder3D() | PlanarOrientedCylinder3D()) if not isinstance(anno_ts, PlanarBbox3D_pin):
                                anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarPolyline3D() | PlanarPolygon3D():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarOccSdfBev():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['sdf'].append(annotation_tensor['sdf'])
                                anno_batch_dict[transformable.name]['height'].append(annotation_tensor['height'])
                            case PlanarPolyline3D_laneline() | PlanarPolygon3D():
                                if 'seg' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                if 'reg' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                                if 'lanecls' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['lanecls'].append(annotation_tensor['lanecls'])
                                if 'laneloc' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['laneloc'].append(annotation_tensor['laneloc'])
                            case PlanarPolyline3D_laneline_mul():
                                if 'lanecls' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['lanecls'].append(annotation_tensor['lanecls'])
                                if 'laneloc' in annotation_tensor:
                                    anno_batch_dict[transformable.name]['laneloc'].append(annotation_tensor['laneloc'])
                            case PlanarParkingSlot3D():
                                anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])  
                                anno_batch_dict[transformable.name]['pts'].append(annotation_tensor['pts']) 
                                # if pin_tensor is generated, append it to the annotation. else use empty tensor from PlanarParkingSlot3D
                                if self.pin_exists(input_dict):
                                    pin_tensor = self.pin_to_tensor(input_dict)
                                    annotation_tensor['pin'] = pin_tensor
                                    anno_batch_dict[transformable.name]['pin'].append(annotation_tensor['pin'])
                                    # print(pin_tensor.max(), pin_tensor.min(), pin_tensor.shape)
                                else:
                                    anno_batch_dict[transformable.name]['pin'].append(annotation_tensor['pin'])  
                            case PlanarBbox3D_pin():
                                continue  # PlanarBbox3D_pin is handled separately in PlanarParkingSlot3D
                            case _:
                                anno_batch_dict[transformable.name].append(annotation_tensor)

        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
                processed_frame_batch['camera_tensors'][cam_id])
        if processed_frame_batch['delta_poses']:
            processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
        for transformable_name, data_batch in anno_batch_dict.items():
            # stack known one-sub-layer tensor_dict
            if isinstance(data_batch, dict):
                for task_name, task_data_batch in data_batch.items():
                    if all(isinstance(data, torch.Tensor) for data in task_data_batch):
                        anno_batch_dict[transformable_name][task_name] = torch.stack(task_data_batch)
            # stack tensor batches
            elif all(isinstance(data, torch.Tensor) for data in data_batch):
                anno_batch_dict[transformable_name] = torch.stack(data_batch)
        return defaultdict2dict(processed_frame_batch)

@MODEL_FEEDERS.register_module()
class FastRayLidarPlanarModelFeeder(BaseModelFeeder):
    # TODO: for sdf_2d, we should mix tensor across frames

    def __init__(self,
                 voxel_feature_config: dict,
                 camera_feature_configs: dict,
                 bilinear_interpolation: bool = True,
                 debug_mode: bool = False):
        super().__init__()
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=self.voxel_feature_config,
            camera_feature_configs=self.camera_feature_configs,
            bilinear_interpolation=bilinear_interpolation
        )
        self.debug_mode = debug_mode

    def process(self, frame_batch: list) -> Union[dict, list]:
        """
        Parameters
        ----------
        frame_batch : list
            list of input_dicts

        Returns
        -------
        dict | list
            processed_frame_batch

        Notes
        -----
        ```
        input_dict = {
            'index_info': index_info,
            'transformables': []
        }
        processed_frame_batch = {
            'index_infos': [index_info, index_info, ...],
            'camera_images': {
                'cam_0': (N, 3, H, W),
                ...
            },
            'camera_lookups': [
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
                {'cam_0': {uu:, Z*X*Y, vv:, Z*X*Y, ...},
            ],
            'delta_poses': [],
            'annotations': {
                'bbox_3d_0': {
                    'cen': (N, 1, X, Y)
                    'seg': (N, C, X, Y)
                    'reg': (N, C, X, Y)
                },
                ...
            },
            ...
        }
        ```
        """
        processed_frame_batch = {
            'index_infos': [],
            'camera_tensors': defaultdict(list),
            'camera_lookups': [],
            'lidar_points': defaultdict(list),
            'annotations': defaultdict(lambda: defaultdict(list)),
            'delta_poses':[]
        }
        if self.debug_mode:
            processed_frame_batch['transformables'] = []
        anno_batch_dict = processed_frame_batch['annotations']
        # rearange input_dict into batches
        for input_dict in frame_batch:
            # batching index info
            processed_frame_batch['index_infos'].append(input_dict['index_info'])
            # append transformables
            if self.debug_mode:
                processed_frame_batch['transformables'].append(input_dict['transformables'])
            # batching transformables
            for transformable in input_dict['transformables'].values():
                match transformable:
                    case CameraImageSet():
                        camera_lookup = self.voxel_lut_gen.generate(transformable)
                        for cam_id in camera_lookup:
                            for lut_key in camera_lookup[cam_id]:
                                camera_lookup[cam_id][lut_key] = torch.tensor(
                                    camera_lookup[cam_id][lut_key])
                        processed_frame_batch['camera_lookups'].append(camera_lookup)
                        for cam_id in transformable.transformables:
                            camera_tensor = transformable.transformables[cam_id].tensor['img']
                            processed_frame_batch['camera_tensors'][cam_id].append(camera_tensor)
                    case LidarPoints():
                        processed_frame_batch['lidar_points']['res_voxels'].append(transformable.tensor['res_voxels'])
                        processed_frame_batch['lidar_points']['res_coors'].append(transformable.tensor['res_coors'])
                        processed_frame_batch['lidar_points']['res_num_points'].append(transformable.tensor['res_num_points'])
                        # processed_frame_batch['lidar_points']['points'].append(transformable.tensor['points'])
                    case EgoPoseSet():
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
                        processed_frame_batch['delta_poses'].append(delta_T)
                    case Bbox3D() | Polyline3D() | ParkingSlot3D() | OccSdfBev():
                        annotation_tensor = transformable.tensor
                        anno_ts = transformable.tensor_smith
                        match anno_ts:
                            case (PlanarBbox3D() | PlanarSquarePillar()
                                  | PlanarCylinder3D() | PlanarOrientedCylinder3D()
                                  | PlanarParkingSlot3D()):
                                anno_batch_dict[transformable.name]['cen'].append(annotation_tensor['cen'])
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarPolyline3D() | PlanarPolygon3D():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['reg'].append(annotation_tensor['reg'])
                            case PlanarOccSdfBev():
                                anno_batch_dict[transformable.name]['seg'].append(annotation_tensor['seg'])
                                anno_batch_dict[transformable.name]['sdf'].append(annotation_tensor['sdf'])
                                anno_batch_dict[transformable.name]['height'].append(annotation_tensor['height'])
                            case _:
                                anno_batch_dict[transformable.name].append(annotation_tensor)

        # tensorize batches
        for cam_id in processed_frame_batch['camera_tensors']:
            processed_frame_batch['camera_tensors'][cam_id] = torch.stack(
                processed_frame_batch['camera_tensors'][cam_id])
        if 'delta_poses' in processed_frame_batch['delta_poses']:
            processed_frame_batch['delta_poses'] = torch.stack(processed_frame_batch['delta_poses'])
        for transformable_name, data_batch in anno_batch_dict.items():
            # stack known one-sub-layer tensor_dict
            if isinstance(data_batch, dict):
                for task_name, task_data_batch in data_batch.items():
                    if all(isinstance(data, torch.Tensor) for data in task_data_batch):
                        anno_batch_dict[transformable_name][task_name] = torch.stack(task_data_batch)
            # stack tensor batches
            elif all(isinstance(data, torch.Tensor) for data in data_batch):
                anno_batch_dict[transformable_name] = torch.stack(data_batch)
        return defaultdict2dict(processed_frame_batch)


@MODEL_FEEDERS.register_module()
class NuscenesFastRayPlanarModelFeeder(FastRayPlanarModelFeeder):
    def process(self, frame_batch: list) -> Union[dict, list]:
        processed_frame_batch = super().process(frame_batch)
        processed_frame_batch.update(sample_token=[], dictionaries=[], ego_poses=[])
        
        for input_dict in frame_batch:
            processed_frame_batch['sample_token'].append(input_dict["transformables"]["sample_token"])
            processed_frame_batch['dictionaries'].append({k: t.dictionary for k, t in input_dict["transformables"].items() if isinstance(t, Bbox3D)})
            processed_frame_batch['ego_poses'].append(input_dict["transformables"]["ego_poses"])
        
        return processed_frame_batch
