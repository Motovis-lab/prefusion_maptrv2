from typing import List, Tuple, Dict
import random
import copy
import inspect
import functools
import abc
from pathlib import Path

import cv2
import mmcv
import torch
import numpy as np

import virtual_camera as vc
# import albumentations as AT

from scipy.spatial.transform import Rotation

from .utils import (
    expand_line_2d, _sign, INF_DIST,
    vec_point2line_along_direction, 
    dist_point2line_along_direction,
    get_cam_type,
    VoxelLookUpTableGenerator
)


from prefusion.registry import TRANSFORMS


def transform_method(func):
    func.is_transform_method = True
    return func


class Transformable:
    """
    Base class for all transformables. 
    It is not a abstract class, because it on one hand provides the full set of transform methods and provide default implementation on the other hand.
    The only purpose of its direct subclasses CameraTransformable and SpatialTransformable is to ensure some transform methods must be implemented.
    """

    def __init__(self, *args, **kwargs):
        pass
    
    @transform_method
    def at_transform(self, func_name, **kwargs):
        return self

    @transform_method
    def adjust_brightness(self, brightness=1, **kwargs):
        return self
    
    @transform_method
    def adjust_saturation(self, saturation=1, **kwargs):
        return self
    
    @transform_method
    def adjust_contrast(self, contrast=1, **kwargs):
        return self
    
    @transform_method
    def adjust_hue(self, hue=1, **kwargs):
        return self
    
    @transform_method
    def adjust_sharpness(self, sharpness=1, **kwargs):
        return self
    
    @transform_method
    def posterize(self, bits=8, **kwargs):
        return self
    
    @transform_method
    def auto_contrast(self, **kwargs):
        return self
    
    @transform_method
    def imequalize(self, **kwargs):
        return self
    
    @transform_method
    def solarize(self, **kwargs):
        return self
    
    @transform_method
    def sobelize(self, **kwargs):
        return self
    
    @transform_method
    def gaussian_blur(self, **kwargs):
        return self
    
    @transform_method
    def channel_shuffle(self, **kwargs):
        return self
    
    @transform_method
    def set_intrinsic_param(self, **kwargs):
        return self
    
    @transform_method
    def render_intrinsic(self, **kwargs):
        return self
    
    @transform_method
    def set_extrinsic_param(self, **kwargs):
        return self
    
    @transform_method
    def render_extrinsic(self, **kwargs):
        return self
    
    @transform_method
    def flip_3d(self, **kwargs):
        return self
    
    @transform_method
    def rotate_3d(self, **kwargs):
        return self

    @transform_method
    def scale_3d(self, **kwargs):
        return self

    def to_tensor(self, tensor_smith, **kwargs):
        return tensor_smith.to_tensor(self)
        


class CameraTransformable(Transformable, metaclass=abc.ABCMeta):
    @transform_method
    @abc.abstractmethod
    def set_intrinsic_param(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def render_intrinsic(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def set_extrinsic_param(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def render_extrinsic(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def flip_3d(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def rotate_3d(self, **kwargs):
        pass


class SpatialTransformable(Transformable, metaclass=abc.ABCMeta):
    @transform_method
    @abc.abstractmethod
    def flip_3d(self, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def rotate_3d(self, **kwargs):
        pass


''' albumentations
AdvancedBlur
Blur
CLAHE
ChannelDropout
ChannelShuffle
ChromaticAberration
Defocus
Emboss
Equalize
FDA
FancyPCA
FromFloat
GaussNoise
GaussianBlur
GlassBlur
HistogramMatching
HueSaturationValue
ISONoise
ImageCompression
InvertImg
MedianBlur
MotionBlur
MultiplicativeNoise
Normalize
PixelDistributionAdaptation
Posterize
RGBShift
RandomBrightnessContrast
RandomFog
RandomGamma
RandomGravel
RandomRain
RandomShadow
RandomSnow
RandomSunFlare
RandomToneCurve
RingingOvershoot
Sharpen
Solarize
Spatter
Superpixels
TemplateTransform
ToFloat
ToGray
ToRGB
ToSepia
UnsharpMask
ZoomBlur
'''

class TransformableSet(Transformable):
    """A set of transformables of the same type. A TransformableSet is also a Transformable."""
    transformable_cls = Transformable  # each subclass of TransformableSet should have its own transformable_cls for protection purpose.

    def __init__(self, transformables: dict):
        self.transformables = transformables
        self.validate_transformable_class()
    
    def validate_transformable_class(self):
        if not all([isinstance(t, self.transformable_cls) for tid, t in self.transformables.items()]):
            raise TypeError(f"transformables of {self.__class__.__name__} should all be of type {self.transformable_cls.__class__.__name__}, but got {[type(t) for t in self.transformables]}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(transformables={self.transformables})"
    
    def _apply_transform(self, method_name, *args, **kwargs):
        for t_id, transformable in self.transformables.items():
            transformable_concerned_args = [arg[t_id] if isinstance(arg, dict) and t_id in arg else arg for arg in args]
            transformable_concerned_kwargs = {k: kwargs[k][t_id] if isinstance(kwargs[k], dict) and t_id in kwargs[k] else kwargs[k] for k in kwargs}
            getattr(transformable, method_name)(*transformable_concerned_args, **transformable_concerned_kwargs)
        return self
    
    def __getattribute__(self, name):
        if name == "_apply_transform":
            return super().__getattribute__("_apply_transform")
        method_or_attr = super().__getattribute__(name)
        if inspect.ismethod(method_or_attr) and hasattr(method_or_attr, 'is_transform_method'):
            return functools.partial(self._apply_transform, name)
        return method_or_attr

    def to_tensor(self, tensor_smith, **kwargs):
        return {tid: t.to_tensor(tensor_smith) for tid, t in self.transformables.items()}


class CameraImage(CameraTransformable):
    def __init__(self, 
        cam_id: str, 
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: np.ndarray,
    ):
        """Image data modeled by specific camera model.

        Parameters
        ----------
        cam_id : str
            camera id
        cam_type : str
            camera type, choices: ['FisheyeCamera', 'PerspectiveCamera']
        img : np.ndarray
            the of image (pixel data), of shape (H, W, C), presume color_channel:=RGB
        ego_mask : np.ndarray
            the mask of ego car (pixel data), of shape (H, W)
        extrinsic : Tuple[np.ndarray, np.ndarray]
            the extrinsic params of this camera, including (R, t), R is of shape (3, 3) and t is of shape(3,)
        intrinsic : np.ndarray
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params


        - \<fast_ray_LUT\> = {
            uu: uu, 
            vv: vv, 
            dd: dd, 
            valid_map: valid_map,
            valid_map_sampled: valid_map_sampled,
            norm_density_map: norm_density_map
        }
        """
        super().__init__()
        assert cam_type in ['FisheyeCamera', 'PerspectiveCamera']
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = list(p.copy() for p in extrinsic)
        self.intrinsic = intrinsic.copy()


    # def at_transform(self, func_name, **kwargs):
    #     func = getattr(AT, func_name)(**kwargs)
    #     self.img = func(image=self.img)['image']
    #     return self

    def adjust_brightness(self, brightness=1, **kwargs):
        self.img = mmcv.adjust_brightness(self.img, factor=brightness)
        return self
    
    def adjust_saturation(self, saturation=1, **kwargs):
        self.img = mmcv.adjust_color(self.img, alpha=saturation)
        return self
    
    def adjust_contrast(self, contrast=1, **kwargs):
        self.img = mmcv.adjust_contrast(self.img, factor=contrast)
        return self
    
    def adjust_hue(self, hue=1, **kwargs):
        self.img = mmcv.adjust_hue(self.img, hue_factor=hue)
        return self
    
    def adjust_sharpness(self, sharpness=1, **kwargs):
        self.img = mmcv.adjust_sharpness(self.img, factor=sharpness)
        return self
    
    def posterize(self, bits=8, **kwargs):
        self.img = mmcv.posterize(self.img, bits)
        return self
    
    def auto_contrast(self, **kwargs):
        self.img = mmcv.auto_contrast(self.img)
        return self
    
    def imequalize(self, **kwargs):
        self.img = mmcv.imequalize(self.img)
        return self
    
    def solarize(self, **kwargs):
        self.img = mmcv.solarize(self.img)
        return self
    
    def channel_shuffle(self, order=[0, 1, 2], **kwargs):
        assert len(order) == self.img.shape[2]
        self.img = self.img[..., order]
        return self
    
    def set_intrinsic_param(self, percentile=0.5, **kwargs):
        # TODO: may need to move the random operation outside of the transform method.
        cx, cy, fx, fy, *distortion_params = self.intrinsic
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.intrinsic = [cx_, cy_, fx_, fy_, *distortion_params]
        return self

    def set_extrinsic_param(self, angle=1, translation=0.05, **kwargs):
        # TODO: may need to move the random operation outside of the transform method.
        R, t = self.extrinsic
        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-angle, angle),
             random.uniform(-angle, angle),
             random.uniform(-angle, angle)],
            degrees=True
        )
        del_t = np.array([
            random.uniform(-translation, translation),
            random.uniform(-translation, translation),
            random.uniform(-translation, translation)
        ])
        R_ = del_R @ R
        t_ = t + del_t
        self.extrinsic = (R_, t_)
        return self

    def render_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.intrinsic), 'invalid intrinsic params'
        resolution_old = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution_old,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        if len(intrinsic) < len(self.intrinsic):
            intrinsic_new = list(intrinsic) + list(self.intrinsic[len(intrinsic):])
        else:
            intrinsic_new = intrinsic
        camera_new = camera_class(
            resolution,
            self.extrinsic,
            intrinsic_new
        )
        self.img, self.ego_mask = vc.render_image(self.img, camera_old, camera_new)
        self.intrinsic = intrinsic_new
        
        return self
    
    def render_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        R, t = self.extrinsic
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.intrinsic
        )
        self.img, self.ego_mask = vc.render_image(self.img, camera_old, camera_new)
        self.extrinsic = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.extrinsic[0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.extrinsic[1] @ flip_mat.T
        self.extrinsic = (R_new, t_new)
        self.intrinsic[0] = self.img.shape[1] - 1 - self.intrinsic[0]
        self.img = np.array(self.img[:, ::-1])
        self.ego_mask = np.array(self.ego_mask[:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.extrinsic
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.extrinsic = (R_new, t_new)
        return self



class CameraImageSet(TransformableSet):
    transformable_cls = CameraImage


class LidarPoints(Transformable):
    def __init__(self, positions: np.ndarray, intensity: np.ndarray): 
        """Lidar points

        Parameters
        ----------
        positions : np.ndarray
            of shape (N, 3), usually in ego-system
        intensity : np.ndarray
            of shape (N, 1)
        """
        self.positions = positions.copy()
        self.intensity = intensity.copy()

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # here points is a row array
        self.positions = self.positions @ flip_mat.T
        
        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        self.positions = self.positions @ rmat.T
        return self


class CameraSegMask(CameraTransformable):
    """
    - self.data = {
        'img': <seg_arr>,
        'ego_mask': <arr>,
        'cam_type': < 'FisheyeCamera' | 'PerspectiveCamera' >
        'extrinsic': (R, t),
        'intrinsic': [cx, cy, fx, fy, *distortion_params]
    }
    """

    def __init__(self, 
        cam_id: str,
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: np.ndarray,
        dictionary: dict
    ):
        """Segmentation Mask data modeled by specific camera model.

        Parameters
        ----------
        cam_id : str
            camera id
        cam_type : str
            camera type, choices: ['FisheyeCamera', 'PerspectiveCamera']
        img : np.ndarray
            the of segmentation mask image (pixel data), of shape (H, W, C), presume color_channel:=RGB
        ego_mask : np.ndarray
            the mask of ego car (pixel data), of shape (H, W)
        extrinsic : Tuple[np.ndarray, np.ndarray]
            the extrinsic params of this camera, including (R, t), R is of shape (3, 3) and t is of shape(3,)
        intrinsic : np.ndarray
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params
        dictionary: dict
            dictionary store class infomation of different channels
        """
        super().__init__()
        assert cam_type in ["FisheyeCamera", "PerspectiveCamera"]
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = list(p.copy() for p in extrinsic)
        self.intrinsic = intrinsic.copy()
        self.dictionary = dictionary.copy()

    def set_intrinsic_param(self, percentile=0.5, **kwargs):
        cx, cy, fx, fy, *distortion_params = self.intrinsic
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.intrinsic = [cx_, cy_, fx_, fy_, *distortion_params]
        return self    

    def set_extrinsic_param(self, angle=1, translation=0.05, **kwargs):
        R, t = self.extrinsic
        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-angle, angle),
             random.uniform(-angle, angle),
             random.uniform(-angle, angle)],
            degrees=True
        )
        del_t = np.array([
            random.uniform(-translation, translation),
            random.uniform(-translation, translation),
            random.uniform(-translation, translation)
        ])
        R_ = del_R @ R
        t_ = t + del_t
        self.extrinsic = (R_, t_)
        return self
    

    def render_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.intrinsic), 'invalid intrinsic params'
        resolution_old = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution_old,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        if len(intrinsic) < len(self.intrinsic):
            intrinsic_new = list(intrinsic) + list(self.intrinsic[len(intrinsic):])
        else:
            intrinsic_new = intrinsic
        camera_new = camera_class(
            resolution,
            self.extrinsic,
            intrinsic_new
        )
        self.img, self.ego_mask = vc.render_image(self.img, camera_old, camera_new)
        self.intrinsic = intrinsic_new
        
        return self
    
    def render_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        R, t = self.extrinsic
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.intrinsic
        )
        self.img, self.ego_mask = vc.render_image(
            self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
        )
        self.extrinsic = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.extrinsic[0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.extrinsic[1] @ flip_mat.T
        self.extrinsic = (R_new, t_new)
        self.intrinsic[0] = self.img.shape[1] - 1 - self.intrinsic[0]
        self.img = np.array(self.img[:, ::-1])
        self.ego_mask = np.array(self.ego_mask[:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.extrinsic
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.extrinsic = (R_new, t_new)
        return self


class CameraSegMaskSet(TransformableSet):
    transformable_cls = CameraSegMask


class CameraDepth(CameraTransformable):
    def __init__(self, 
        cam_id: str,
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: np.ndarray,
        depth_mode: str,
    ):
        """Depth data modeled by specific camera model.

        Parameters
        ----------
        cam_id : str
            camera id
        cam_type : str
            camera type, choices: ['FisheyeCamera', 'PerspectiveCamera']
        img : np.ndarray
            the of segmentation mask image (pixel data), of shape (H, W, C), presume color_channel:=RGB
        ego_mask : np.ndarray
            the mask of ego car (pixel data), of shape (H, W)
        extrinsic : Tuple[np.ndarray, np.ndarray]
            the extrinsic params of this camera, including (R, t), R is of shape (3, 3) and t is of shape(3,)
        intrinsic : np.ndarray
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params
        depth_mode: str
            the mode of depth, choices: ['z' or 'd']
            ('z': depth in z axis of camera coordinate, 'd': depth in distance of point to camera optical point)
        """
        super().__init__()
        assert cam_type in ["FisheyeCamera", "PerspectiveCamera"]
        assert depth_mode in ["z", "d"]
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = list(p.copy() for p in extrinsic)
        self.intrinsic = intrinsic.copy()
        self.depth_mode = depth_mode


    def set_intrinsic_param(self, percentile=0.5, **kwargs):
        cx, cy, fx, fy, *distortion_params = self.intrinsic
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.intrinsic = [cx_, cy_, fx_, fy_, *distortion_params]
        return self
    

    def set_extrinsic_param(self, angle=1, translation=0.05, **kwargs):
        R, t = self.extrinsic
        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-angle, angle),
             random.uniform(-angle, angle),
             random.uniform(-angle, angle)],
            degrees=True
        )
        del_t = np.array([
            random.uniform(-translation, translation),
            random.uniform(-translation, translation),
            random.uniform(-translation, translation)
        ])
        R_ = del_R @ R
        t_ = t + del_t
        self.extrinsic = (R_, t_)
        return self
    

    def render_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.intrinsic), 'invalid intrinsic params'
        resolution_old = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution_old,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        if len(intrinsic) < len(self.intrinsic):
            intrinsic_new = list(intrinsic) + list(self.intrinsic[len(intrinsic):])
        else:
            intrinsic_new = intrinsic   
        camera_new = camera_class(
            resolution,
            self.extrinsic,
            intrinsic_new
        )
        self.img, self.ego_mask = vc.render_image(
            self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
        )
        self.intrinsic = intrinsic_new
        
        return self
    
    def render_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        R, t = self.extrinsic
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.intrinsic
        )
        # TODO: get real points from depth then remap to image
        if self.depth_mode == 'd':
            self.img, self.ego_mask = vc.render_image(
                self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
            )
        elif self.depth_mode == 'z':
            raise NotImplementedError
        self.extrinsic = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.extrinsic[0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.extrinsic[1] @ flip_mat.T
        self.extrinsic = (R_new, t_new)
        self.intrinsic[0] = self.img.shape[1] - 1 - self.intrinsic[0]
        self.img = np.array(self.img[:, ::-1])
        self.ego_mask = np.array(self.ego_mask[:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.extrinsic
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.extrinsic = (R_new, t_new)
        return self



class CameraDepthSet(TransformableSet):
    transformable_cls = CameraDepth


class Bbox3D(SpatialTransformable):
    def __init__(self, boxes: List[dict], dictionary: dict):
        """
        Parameters
        ----------
        boxes : List[dict]
            a list of boxes. Each box is a dict having the following format:
            boxes[0] = {
                'class': 'class.vehicle.passenger_car',
                'attr': {'attr.time_varying.object.state': 'attr.time_varying.object.state.stationary',
                        'attr.vehicle.is_trunk_open': 'attr.vehicle.is_trunk_open.false',
                        'attr.vehicle.is_door_open': 'attr.vehicle.is_door_open.false'},
                'size': [4.6486, 1.9505, 1.5845],
                'rotation': array([[ 0.93915682, -0.32818596, -0.10138267],
                                [ 0.32677338,  0.94460343, -0.03071667],
                                [ 0.1058472 , -0.00428138,  0.99437319]]),
                'translation': array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # NOTE: it is a column vector
                'track_id': '10035_0', # NOT USED
                'velocity': array([[0.], [0.], [0.]]) # NOTE: it is a column vector
            }
        dictionary : dict
            Example dictionary: {
                'branch_0': {
                    'classes': ['car', 'bus', 'pedestrain', ...],
                    'attrs': []
                }
                'branch_1': {
                    'classes': [],
                    'attrs': []
                }
                ...
            }
        """
        self.boxes = boxes.copy()
        self.dictionary = dictionary.copy()
        self.remove_boxes_not_defined_in_dictionary()

    def remove_boxes_not_defined_in_dictionary(self, **kwargs):
        full_set_of_classes = {c for branch in self.dictionary.values() for c in branch['classes']}
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[i]['class'] not in full_set_of_classes:
                del self.boxes[i]

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        for box in self.boxes:
            box['rotation'] = flip_mat @ box['rotation'] @ flip_mat_self.T
            # here translation is a row array
            box['translation'] = flip_mat @ box['translation']
            box['velocity'] = flip_mat @ box['velocity']
        
        # TODO: flip classname for arrows
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for box in self.boxes:
            box['rotation'] = rmat @ box['rotation']
            box['translation'] = rmat @ box['translation']
            box['velocity'] = rmat @ box['velocity']
        return self
    


class BboxBev(Bbox3D):
    pass


class Cylinder3D(Bbox3D):
    pass


class OrientedCylinder3D(Bbox3D):
    pass


class Square3D(Bbox3D):
    pass


class Polyline3D(SpatialTransformable):
    '''
    - self.data = {
        'elements:' [element, element, ...],
        'tensor': <tensor>
      }
    - element = {
        'class': 'class.road_marker.lane_line',
        'attr': <dict>,
        'points': <N x 3 array>
    }
    '''
    def __init__(self, data: list, dictionary: dict):
        self.dictionary = dictionary
        # filter elements by dictionary
        available_elements = []
        for branch in dictionary:
            available_elements.extend(dictionary[branch]['classes'])
        self.data = {'elements': []}
        for element in data:
            if element['class'] in available_elements:
                self.data['elements'].append(element)

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # here points is a row array
        for element in self.data['elements']:
            element['points'] = element['points'] @ flip_mat.T
        
        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for element in self.data['elements']:
            element['points'] = element['points'] @ rmat.T
        return self


class Polygon3D(Polyline3D):
    pass


class ParkingSlot3D(Polyline3D):

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        # here points is a row array
        for element in self.data['elements']:
            element['points'] = flip_mat_self @ element['points'] @ flip_mat.T
        
        return self



class Trajectory(SpatialTransformable):
    '''
    - self.data = {
        'elements': [element, element, ...],
        'tensor': <tensor>
    }
    - element = [(R, t), (R, t), ...]
    '''
    def __init__(self, data: list):
        self.data = {'elements': data}

    def flip_3d(self, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, **kwargs):
        raise NotImplementedError



class SegBev(SpatialTransformable):
    def flip_3d(self, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, **kwargs):
        raise NotImplementedError


class OccSdfBev(SpatialTransformable):

    '''
    self.data = {
        'src_view_range': [back, front, right, left, bottom, up], # in ego system
        'occ': <N x H x W>, # H <=> (xmin, xmax), W <=> (ymin, ymax)
        'sdf': <1 x H x W>,
        'height': <1 x H x W>,
        'mask': <1 x H x W>,
    }
    
    back, front, right, left, bottom, up  
    ||    ||     ||     ||    ||      ||
    xmin, xmax,  ymin,  ymax, zmin,   zmax

    bev: backward-right-up (H, W, Z)
    ego: x-y-z, forward-left-up
    '''
    def __init__(self, data: dict, dictionary: dict):
        super().__init__(data)
        self.dictionary = dictionary
        if 'mask' not in self.data:
            self.data['mask'] = np.ones(self.data['sdf'].shape, dtype=np.uint8)
        self._bev_shape = self.data['occ'].shape[1:]
        self._ego_points = self._unproject_bev_to_ego()


    def _unproject_bev_to_ego(self):
        H, W = self._bev_shape
        fx = H / (self.data['src_view_range'][0] - self.data['src_view_range'][1])
        fy = W / (self.data['src_view_range'][2] - self.data['src_view_range'][3])
        cx = - self.data['src_view_range'][1] * fx - 0.5
        cy = - self.data['src_view_range'][3] * fy - 0.5
        self._bev_intrinsic = [cx, cy, fx, fy]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        xx = (vv - cx) / fx
        yy = (uu - cy) / fy
        zz = self.data['height'][0]
        # coloum points
        return np.stack([xx, yy, zz], axis=0).reshape(3, -1)
    

    def _project_ego_to_bev(self, ego_points):
        xx, yy, _ = ego_points
        cx, cy, fx, fy = self._bev_intrinsic
        vv_ = xx * fx + cx
        uu_ = yy * fy + cy
        return uu_, vv_


    def flip_3d(self, flip_mat, **kwargs):
        # 1. get ego coordinates from bev
        # 2. apply flip_mat
        # 3. project to bev
        # 4. remap
        assert flip_mat[2, 2] == 1, 'up-down flipping is unnecessary!'
        flipped_points = flip_mat @ self._ego_coords
        uu_, vv_ = self._project_ego_to_bev(flipped_points)
        
        self.data['occ'] = cv2.remap(
            self.data['occ'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_NEAREST
        )
        self.data['sdf'] = cv2.remap(
            self.data['sdf'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_LINEAR
        )
        self.data['height'] = cv2.remap(
            self.data['height'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_LINEAR
        )
        self.data['mask'] = cv2.remap(
            self.data['mask'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_NEAREST
        )
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        # 1. get ego coordinates from bev
        # 2. apply rotation
        # 3. project to bev
        rotated_points = rmat @ self._ego_coords
        uu_, vv_ = self._project_ego_to_bev(rotated_points)
        
        self.data['occ'] = cv2.remap(
            self.data['occ'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_NEAREST
        )
        self.data['sdf'] = cv2.remap(
            self.data['sdf'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_LINEAR
        )
        self.data['height'] = cv2.remap(
            self.data['height'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_LINEAR
        )
        self.data['mask'] = cv2.remap(
            self.data['mask'], 
            uu_.reshape((1, *self._bev_shape)),
            vv_.reshape((1, *self._bev_shape)),
            interpolation=cv2.INTER_NEAREST
        )
        return self


class OccSdf3D(SpatialTransformable):
    def flip_3d(self, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, **kwargs):
        raise NotImplementedError

#--------------------------------#




class Transform:
    '''
    Basic class for Transform.
    '''
    def __init__(self, scope="frame"):
        assert scope.lower() in ["frame", "batch", "group"]
        self.scope = scope.lower()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def random_transform_class_factory(cls_name, transform_func):
    """
    pipeline = [
        dict(
            type='RandomTransformSequence',
            transforms=[
                dict(
                    type='RandomBrightness',
                    value_random_definition=dict(brightness=dict(type=float, range=[0.5, 1.5])),
                    scope='frame',
                    prob=0.5
                ),
                dict(
                    type='RandomSaturation',
                    value_random_definition={'type': 'uniform', 'range': [0.5, 1.5]},
                    prob=0.5
                )
            ]
        )
    ]
    """
    def __init__(self, *, prob: float = 0.0, param_randomization_rules: dict = None, scope: str = "frame", **kwargs):
        """Initialize a Transform object.

        Parameters
        ----------
        prob : float, optional
            the happening probability of this Transform, value range [0, 1], by default 0.0
        param_randomization_rules : dict
            Definition of how the param should be randomized, e.g. {"param_name": {"type": "float", "range": [0, 1]}.
            Current supported types includes: 'float', 'int' and 'enum'. For 'float' and 'int', use "range": [xxx, yyy]; for 'enum', use choices [a, b, c]
        scope : str, optional
            the scope of the Transform, by default "frame"
        """
        Transform.__init__(self, scope=scope)
        self.prob = prob
        self.param_randomization_rules = param_randomization_rules
        self.kwargs = kwargs
        self.validate_param_randomization_rules()
    
    def validate_param_randomization_rules(self):
        assert isinstance(self.param_randomization_rules, dict), f"param_randomization_rules should be a dict. But {self.param_randomization_rules} is given."
        for _, rule in self.param_randomization_rules.items():
            assert rule["type"] in ["float", "int", "enum"], f"Only 'float', 'int' and 'enum' are valid types for a rule. But {rule['type']} is given."
            if rule["type"] == "enum":
                assert "choices" in rule, "choices should be used along with type: enum."
            else:
                assert "range" in rule, "range should be used along with type: float or int."

    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        if random.random() > self.prob:
            return list(transformables)
        # TODO: implement the randomization for params
        return [None if i is None else getattr(i, transform_func)(**self.kwargs) for i in transformables]

    return type(cls_name, (Transform,), {"__init__": __init__, "__call__": __call__})


def deterministic_transform_class_factory(cls_name, transform_func):
    def __init__(self, scope="frame", **kwargs):
        Transform.__init__(self, scope=scope)
        self.kwargs = kwargs

    def __call__(self, *transformables, **kwargs):
        return [None if i is None else getattr(i, transform_func)(**self.kwargs) for i in transformables]

    return type(cls_name, (Transform,), {"__init__": __init__, "__call__": __call__})



RandomBrightness = random_transform_class_factory("RandomBrightness", "adjust_brightness")
RandomSaturation = random_transform_class_factory("RandomSaturation", "adjust_saturation")
RandomContrast = random_transform_class_factory("RandomContrast", "adjust_contrast")
RandomHue = random_transform_class_factory("RandomHue", "adjust_hue")
RandomSharpness = random_transform_class_factory("RandomSharpness", "adjust_sharpness")
RandomPosterize = random_transform_class_factory("RandomPosterize", "posterize")
RandomChannelShuffle = random_transform_class_factory("RandomChannelShuffle", "channel_shuffle")
RandomAutoContrast = random_transform_class_factory("RandomAutoContrast", "auto_contrast")
RandomSolarize = random_transform_class_factory("RandomSolarize", "solarize")
RandomImEqualize = random_transform_class_factory("RandomImEqualize", "imequalize")
RandomSetIntrinsicParam = random_transform_class_factory("RandomSetIntrinsicParam", "set_intrinsic_param")
RandomSetExtrinsicParam = random_transform_class_factory("RandomSetExtrinsicParam", "set_extrinsic_param")



#######################
# Customed Transforms #
#######################



class RandomChooseOneTransform(Transform):
    def __init__(self, transforms, *, prob=0.5, transform_probs=None, scope="group"):
        super().__init__(scope=scope)
        self.transforms = transforms
        if transform_probs is not None:
            assert len(transforms) == len(transform_probs)
        self.transform_probs = transform_probs
        self.prob = prob

    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        if random.random() <= self.prob:
            transform = random.choices(
                population=self.transforms,
                weights=self.transform_probs,
                k=1
            )[0]
            transform(*transformables, **kwargs)
        return transformables



class RandomTransformSequence(Transform):
    def __init__(self, transforms, *, scope='frame') -> None:
        self.transforms = copy.deepcopy(transforms)
        self.scope = scope
    
    def __call__(self, *transformables, seeds: dict = None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        random.shuffle(self.transforms)
        for transform in self.transforms:
            transform(*transformables, seeds=seeds, **kwargs)
        return transformables


# reimplement by passing transforms as arguments
class RandomImageISP(Transform):
    def __init__(self, *, 
                 adjust_brightness={"prob": 0.5, "range": (0.5, 2.0)},
                 adjust_saturation={"prob": 0.5, "range": (0.0, 2.0)},
                 adjust_contrast={"prob": 0.5, "range": (0.5, 2.0)},
                 adjust_hue={"prob": 0.5, "range": (-0.5, 0.5)},
                 adjust_sharpness={"prob": 0.5, "range": (0.0, 2.0)},
                 posterize={"prob": 0.5, "bits": (4, 8)},
                 channel_shuffle={"prob": 0.5},
                 auto_contrast={"prob": 0.2},
                 solarize={"prob": 0.01},
                 imequalize={"prob": 0.1},
                 random_sequence=True,
                 scope="frame",  **kwargs):
        super().__init__(scope=scope)
        transforms = list(inspect.signature(self.__init__).parameters.keys())[:-3]
        self.random_sequence = random_sequence
        self.sequence = []
        for transform in transforms:
            self.sequence.append({transform: eval(f"{transform}")})
        self.kwargs = kwargs
    
    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        
        if self.random_sequence:
            random.shuffle(self.sequence)
        
        sequence = {}
        for transform in self.sequence:
            sequence.update(transform)
        
        for transformable in transformables:
            if transformable is not None:
                for transform_func in sequence:
                    # print(transform_func)
                    prob = sequence[transform_func]['prob']
                    if random.random() < prob:
                        if 'range' in sequence[transform_func]:
                            random_value = random.uniform(*sequence[transform_func]['range'])
                            getattr(transformable, transform_func)(random_value, **kwargs)
                        elif 'bits' in sequence[transform_func]:
                            random_value = random.randint(*sequence[transform_func]['bits'])
                            getattr(transformable, transform_func)(random_value, **kwargs)
                        else:
                            getattr(transformable, transform_func)(**kwargs)
        
        return transformables



class RandomImageTransformAT(Transform):
    pass



class RandomImageOmit(Transform):
    pass


class RenderIntrinsic(Transform):
    
    def __init__(self, resolutions, intrinsics='auto', scope="frame"):
        '''
        resolutions: {<cam_id>: (W, H), ...}
        intrinsics: {<cam_id>: (cx, cy, fx, fy, ...), ...}
        '''
        super().__init__(scope=scope)
        self.cam_ids = list(resolutions.keys())
        self.resolutions = resolutions
        if intrinsics == 'auto':
            self.intrinsics = {}
            for cam_id in self.cam_ids:
                W, H = self.resolutions[cam_id]
                cx = (W - 1) / 2
                cy = (H - 1) / 2
                if get_cam_type(cam_id) in ['FisheyeCamera']:
                    fx = fy = W / 4
                else:
                    fx = fy = W / 2
                intrinsic = [cx, cy, fx, fy]
                self.intrinsics[cam_id] = intrinsic
        else:
            self.intrinsics = intrinsics
    
    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for cam_id, t in transformable.transformables.items():
                    if cam_id in self.cam_ids:
                        t.render_intrinsic(self.resolution[cam_id], self.intrinsic[cam_id])
            elif isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                t.render_intrinsic(self.resolution[cam_id], self.intrinsic[cam_id])
        return transformables



class RenderExtrinsic(Transform):
    
    def __init__(self, del_rotations, scope="frame"):
        '''
        del_rotations: {
            <cam_id>: [<ang>, <ang>, <ang>], # ego x-y-z euler angles
            ...
        }
        '''
        super().__init__(scope=scope)
        self.cam_ids = list(del_rotations.keys())
        self.del_extrinsics = {}
        for cam_id in self.cam_ids:
            del_R = Rotation.from_euler(
                'xyz', del_rotations[cam_id], degrees=True
            ).as_matrix()
            del_t = np.array([0, 0, 0])
            self.del_extrinsics[cam_id] = (del_R, del_t)
    
    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                if transformable.data['cam_id'] in self.cam_ids:
                    del_extrinsic = self.del_extrinsics[transformable.data['cam_id']]
                    transformable.render_extrinsic(del_extrinsic)
        return transformables



class FastRayLookUpTable(Transform):
    
    def __init__(self, voxel_feature_config, camera_feature_configs, scope="frame"):
        '''
        voxel_feature_config = dict(
            voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
            voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
            ego_distance_max=40,
            ego_distance_step=2
        )
        general_camera_feature_config = dict(
            ray_distance_num_channel=64,
            ray_distance_start=0.25,
            ray_distance_step=0.25,
            feature_downscale=1,
        )
        camera_feature_configs = dict(
            VCAMERA_FISHEYE_FRONT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT_LEFT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK_LEFT=general_camera_feature_config,
            VCAMERA_FISHEYE_LEFT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK=general_camera_feature_config,
            VCAMERA_FISHEYE_BACK=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT_RIGHT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_BACK_RIGHT=general_camera_feature_config,
            VCAMERA_FISHEYE_RIGHT=general_camera_feature_config,
            VCAMERA_PERSPECTIVE_FRONT=general_camera_feature_config
        )
        '''
        super().__init__(scope=scope)
        self.lut_gen = VoxelLookUpTableGenerator(
            voxel_feature_config=voxel_feature_config,
            camera_feature_configs=camera_feature_configs
        )
        self.cam_ids = list(camera_feature_configs.keys())
    
    def __call__(self, *transformables, seeds=None, **kwargs):
        seed = None if seeds is None else seeds[self.scope]
        camera_images = {}
        for transformable in transformables:
            if isinstance(transformable, CameraImage):
                cam_id = transformable.data['cam_id']
                if cam_id in self.cam_ids:
                    camera_images[cam_id] = transformable
        LUT = self.lut_gen.generate(camera_images, seed=seed)
        for cam_id in camera_images:
            camera_images[cam_id].data['fast_ray_LUT'] = LUT[cam_id]
        return transformables



class RandomRenderExtrinsic(Transform):
    def __init__(self, *, prob=0.5, angles=[1, 1, 1], scope="frame", **kwargs):
        super().__init__(scope=scope)
        self.prob = prob
        self.angles = angles

    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        if random.random() > self.prob:
            return list(transformables)
        
        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-self.angles[0], self.angles[0]),
             random.uniform(-self.angles[1], self.angles[1]),
             random.uniform(-self.angles[2], self.angles[2])],
            degrees=True
        ).as_matrix()

        for transformable in transformables:
            if isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                del_extrinsic = (del_R, np.array([0, 0, 0]))
                transformable.render_extrinsic(del_extrinsic)
        return transformables



class RandomRotationSpace(Transform):
    def __init__(self, *, prob=0.5, angles=[2, 2, 10], 
                 prob_inverse_cameras_rotation=0.5, scope="group", **kwargs):
        '''
        angles = [roll, pitch, yaw] in degrees, x-y-z
        '''
        super().__init__(scope=scope)
        self.prob = prob
        self.angles = angles
        self.prob_inverse_cameras_rotation = prob_inverse_cameras_rotation

    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        if random.random() > self.prob:
            return list(transformables)
        
        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-self.angles[0], self.angles[0]),
             random.uniform(-self.angles[1], self.angles[1]),
             random.uniform(-self.angles[2], self.angles[2])],
            degrees=True
        ).as_matrix()

        inverse_cameras_rotation = False
        if random.random() < self.prob_inverse_cameras_rotation:
            inverse_cameras_rotation = True

        for transformable in transformables:
            if transformable is not None:
                transformable.rotate_3d(del_R)
                if inverse_cameras_rotation:
                    del_extrinsic = (del_R.T, np.array([0.0, 0, 0]))
                    transformable.render_extrinsic(del_extrinsic)
        
        return transformables



class RandomMirrorSpace(Transform):
    def __init__(self, *, prob=0.5, flip_mode='Y', scope="group", **kwargs):
        '''
        flip_mode: support X, Y, Z, XY, XZ, YZ, XYZ, however, Z is unnessary
        '''
        super().__init__(scope=scope)
        self.prob = prob
        self.flip_mode = flip_mode
        self.flip_mat = np.eye(3)
        if 'X' in self.flip_mode:
            self.flip_mat[0, 0] = -1
        if 'Y' in self.flip_mode:
            self.flip_mat[1, 1] = -1
        if 'Z' in self.flip_mode:
            self.flip_mat[2, 2] = -1

    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds:
            random.seed(seeds[self.scope])
        if random.random() > self.prob:
            return list(transformables)

        for transformable in transformables:
            if transformable is not None:
                transformable.flip_3d(self.flip_mat)
        


class MirrorTime(Transform):
    pass



class ScaleTime(Transform):
    pass




available_transforms = [
    RandomBrightness,
    RandomSaturation,
    RandomContrast,
    RandomHue,
    RandomSharpness,
    RandomPosterize,
    RandomChannelShuffle,
    RandomAutoContrast,
    RandomSolarize,
    RandomImEqualize,
    RandomImageISP, 
    RenderIntrinsic, 
    RenderExtrinsic, 
    RandomRenderExtrinsic, 
    RandomRotationSpace, 
    RandomMirrorSpace, 
    RandomSetIntrinsicParam, 
    RandomSetExtrinsicParam, 
]

for transform in available_transforms:
    TRANSFORMS.register_module(module=transform)
