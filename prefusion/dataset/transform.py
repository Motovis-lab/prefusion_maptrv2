import random
from copy import deepcopy
import inspect
import functools
import abc
from typing import List, Tuple, Dict, Union, Sequence, TYPE_CHECKING, Any

import cv2
import mmcv
import torch
import numpy as np
import virtual_camera as vc
# import albumentations as AT
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from copious.cv.geometry import Box3d as CopiousBox3d

# from .tensor_smith import TensorSmith
from prefusion.registry import TRANSFORMS, TRANSFORMABLES
from prefusion.dataset.utils import make_seed


if TYPE_CHECKING:
    from .tensor_smith import TensorSmith


def transform_method(func):
    func.is_transform_method = True
    return func


class Transformable:
    """
    Base class for all transformables. 
    It is not a abstract class, because it on one hand provides the full set of transform methods and provide default implementation on the other hand.
    The only purpose of its direct subclasses CameraTransformable and SpatialTransformable is to ensure some transform methods must be implemented.
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @transform_method
    def at_transform(self, func_name, *args, **kwargs):
        return self

    @transform_method
    def adjust_brightness(self, *args, brightness=1, **kwargs):
        return self
    
    @transform_method
    def adjust_saturation(self, *args, saturation=1, **kwargs):
        return self
    
    @transform_method
    def adjust_contrast(self, *args, contrast=1, **kwargs):
        return self
    
    @transform_method
    def adjust_hue(self, *args, hue=1, **kwargs):
        return self
    
    @transform_method
    def adjust_sharpness(self, *args, sharpness=1, **kwargs):
        return self
    
    @transform_method
    def posterize(self, *args, bits=8, **kwargs):
        return self
    
    @transform_method
    def auto_contrast(self, *args, **kwargs):
        return self
    
    @transform_method
    def imequalize(self, *args, **kwargs):
        return self
    
    @transform_method
    def solarize(self, *args, **kwargs):
        return self
    
    @transform_method
    def sobelize(self, *args, **kwargs):
        return self
    
    @transform_method
    def gaussian_blur(self, *args, **kwargs):
        return self
    
    @transform_method
    def channel_shuffle(self, *args, **kwargs):
        return self
    
    @transform_method
    def set_intrinsic_param(self, *args, **kwargs):
        return self
    
    @transform_method
    def render_intrinsic(self, *args, **kwargs):
        return self
    
    @transform_method
    def set_extrinsic_param(self, *args, **kwargs):
        return self
    
    @transform_method
    def render_extrinsic(self, *args, **kwargs):
        return self
    
    @transform_method
    def render_camera(self, *args, **kwargs):
        return self
    
    @transform_method
    def flip_3d(self, *args, **kwargs):
        return self
    
    @transform_method
    def rotate_3d(self, *args, **kwargs):
        return self

    @transform_method
    def scale_3d(self, *args, **kwargs):
        return self

    def to_tensor(self):
        """
        Attributes
        ----------
        tensor : torch.Tensor or Dict[str, torch.Tensor], default None
            A tensor that represents the input to the model. This attribute
            does not necessarily need to be implemented here; it is merely a
            suggestion, considering that in some cases, model inputs may
            require multiple transformables to be combined into a tensor.

        Returns
        -------
        self
            transformable itself.
        """
        if hasattr(self, 'tensor'):
            raise RuntimeError('self.tensor cannot be assigned before self.to_tensor().')
        
        if hasattr(self, 'tensor_smith') and callable(self.tensor_smith):
            self.tensor = self.tensor_smith(self)

        return self
        


class CameraTransformable(Transformable, metaclass=abc.ABCMeta):
    @transform_method
    @abc.abstractmethod
    def set_intrinsic_param(self, *args, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def render_intrinsic(self, *args, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def set_extrinsic_param(self, *args, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def render_extrinsic(self, *args, **kwargs):
        pass

    @transform_method
    @abc.abstractmethod
    def render_camera(self, *args, **kwargs):
        return self
    
    @transform_method
    @abc.abstractmethod
    def flip_3d(self, *args, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def rotate_3d(self, *args, **kwargs):
        pass


class SpatialTransformable(Transformable, metaclass=abc.ABCMeta):
    @transform_method
    @abc.abstractmethod
    def flip_3d(self, *args, **kwargs):
        pass
    
    @transform_method
    @abc.abstractmethod
    def rotate_3d(self, *args, **kwargs):
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

    def __init__(self, name, transformables: dict):
        super().__init__(name)
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

    def to_tensor(self):
        return {tid: t.to_tensor() for tid, t in self.transformables.items()}



class CameraImage(CameraTransformable):
    def __init__(self, 
        name: str,
        cam_id: str, 
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: Union[np.ndarray, list, tuple],
        tensor_smith: "TensorSmith" = None
    ):
        """Image data modeled by specific camera model.

        Parameters
        ----------
        name: str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        cam_id : str
            camera id
        cam_type : str
            camera type, choices: ['FisheyeCamera', 'PerspectiveCamera']
        img : np.ndarray
            the of image (pixel data), of shape (H, W, C), presume color_channel:=BGR
        ego_mask : np.ndarray
            the mask of ego car (pixel data), of shape (H, W)
        extrinsic : Tuple[np.ndarray, np.ndarray]
            the extrinsic params of this camera, including (R, t), R is of shape (3, 3) and t is of shape(3,)
        intrinsic : Union[np.ndarray, list, tuple]
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        assert cam_type in ['FisheyeCamera', 'PerspectiveCamera']
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.tensor_smith = tensor_smith


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
    
    def auto_contrast(self, *args, **kwargs):
        self.img = mmcv.auto_contrast(self.img)
        return self
    
    def imequalize(self, *args, **kwargs):
        self.img = mmcv.imequalize(self.img)
        return self
    
    def solarize(self, *args, **kwargs):
        self.img = mmcv.solarize(self.img)
        return self
    
    def channel_shuffle(self, order=[0, 1, 2], **kwargs):
        assert len(order) == self.img.shape[2]
        self.img = self.img[..., order]
        return self
    def set_intrinsic_param(self, intrinsic: Sequence, **kwargs):
        self.intrinsic = intrinsic
        return self

    def set_extrinsic_param(self, extrinsic: Tuple[np.ndarray, np.ndarray], **kwargs):
        self.extrinsic = extrinsic
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
    
    def render_camera(self, camera_new, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        camera_new.extrinsic = (camera_new.extrinsic[0], self.extrinsic[1])
        self.img, self.ego_mask = vc.render_image(self.img, camera_old, camera_new)
        self.extrinsic = camera_new.extrinsic
        self.intrinsic = camera_new.intrinsic
        self.cam_type = camera_new.__class__.__name__
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a camera is left-right symmetrical, 
        # however, x-axis of camera coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[0, 0] = -1
        R_new = flip_mat @ self.extrinsic[0] @ flip_mat_self.T
        # R_new = flip_mat.T @ self.extrinsic[0]
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


class CameraSegMask(CameraTransformable):

    def __init__(self, 
        name: str,
        cam_id: str,
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: Union[np.ndarray, list, tuple],
        dictionary: dict,
        tensor_smith: "TensorSmith" = None
    ):
        """Segmentation Mask data modeled by specific camera model.

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        cam_id : str
            camera id
        cam_type : str
            camera type, choices: ['FisheyeCamera', 'PerspectiveCamera']
        img : np.ndarray
            the of segmentation mask image (pixel data), of shape (H, W)
        ego_mask : np.ndarray
            the mask of ego car (pixel data), of shape (H, W)
        extrinsic : Tuple[np.ndarray, np.ndarray]
            the extrinsic params of this camera, including (R, t), R is of shape (3, 3) and t is of shape(3,)
        intrinsic : Union[np.ndarray, list, tuple]
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params
        dictionary: dict
            dictionary store class infomation of different values
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        assert cam_type in ["FisheyeCamera", "PerspectiveCamera"]
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        assert dictionary is not None
        self.dictionary = dictionary
        self.tensor_smith = tensor_smith

    def set_intrinsic_param(self, intrinsic: Sequence, **kwargs):
        self.intrinsic = intrinsic
        return self   

    def set_extrinsic_param(self, extrinsic: Tuple[np.ndarray, np.ndarray], **kwargs):
        self.extrinsic = extrinsic
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
        self.img, self.ego_mask = vc.render_image(
            self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
        )
        self.extrinsic = (R_new, t_new)
        
        return self
    
    def render_camera(self, camera_new, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        camera_new.extrinsic = (camera_new.extrinsic[0], self.extrinsic[1])
        self.img, self.ego_mask = vc.render_image(
            self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
        )
        self.extrinsic = camera_new.extrinsic
        self.intrinsic = camera_new.intrinsic
        self.cam_type = camera_new.__class__.__name__
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a camera is left-right symmetrical, 
        # however, x-axis of camera coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[0, 0] = -1
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
        name: str,
        cam_id: str,
        cam_type: str, 
        img: np.ndarray, 
        ego_mask: np.ndarray, 
        extrinsic: Tuple[np.ndarray, np.ndarray], 
        intrinsic: Union[np.ndarray, list, tuple],
        depth_mode: str,
        tensor_smith: "TensorSmith" = None
    ):
        """Depth data modeled by specific camera model.

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
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
        intrinsic : Union[np.ndarray, list, tuple]
            if it's PerspectiveCamera, it contains 4 values: cx, cy, fx, fy
            if it's FisheyeCamera, it contains more values: cx, cy, fx, fy, *distortion_params
        depth_mode: str
            the mode of depth, choices: ['z' or 'd']
            ('z': depth in z axis of camera coordinate, 'd': depth in distance of point to camera optical point)
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        assert cam_type in ["FisheyeCamera", "PerspectiveCamera"]
        assert depth_mode in ["z", "d"]
        self.cam_id = cam_id
        self.cam_type = cam_type
        self.img = img
        self.ego_mask = ego_mask
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.depth_mode = depth_mode
        self.tensor_smith = tensor_smith


    def set_intrinsic_param(self, intrinsic: Sequence, **kwargs):
        self.intrinsic = intrinsic
        return self

    def set_extrinsic_param(self, extrinsic: Tuple[np.ndarray, np.ndarray], **kwargs):
        self.extrinsic = extrinsic
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
        R, t = self.extrinsic
        del_R, del_t = delta_extrinsic
        R_new, t_new = del_R @ R, del_t + t
        if self.depth_mode == 'd':
            camera_class = getattr(vc, self.cam_type)
            camera_old = camera_class(
                resolution,
                self.extrinsic,
                self.intrinsic,
                ego_mask=self.ego_mask
            )
            camera_new = camera_class(
                resolution,
                (R_new, t_new),
                self.intrinsic
            )
            self.img, self.ego_mask = vc.render_image(
                self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
            )
        elif self.depth_mode == 'z':
            # TODO: get real points from depth then remap to image
            raise NotImplementedError
        self.extrinsic = (R_new, t_new)
        
        return self
    
    def render_camera(self, camera_new, **kwargs):
        resolution = self.img.shape[:2][::-1]
        camera_class = getattr(vc, self.cam_type)
        camera_old = camera_class(
            resolution,
            self.extrinsic,
            self.intrinsic,
            ego_mask=self.ego_mask
        )
        camera_new.extrinsic = (camera_new.extrinsic[0], self.extrinsic[1])
        if self.depth_mode == 'd':
            self.img, self.ego_mask = vc.render_image(
                self.img, camera_old, camera_new, interpolation=cv2.INTER_NEAREST
            )
        else:
            # TODO: get real points from depth then remap to image
            raise NotImplementedError
        self.extrinsic = camera_new.extrinsic
        self.intrinsic = camera_new.intrinsic
        self.cam_type = camera_new.__class__.__name__
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a camera is left-right symmetrical, 
        # however, x-axis of camera coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[0, 0] = -1
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


class LidarPoints(SpatialTransformable):
    def __init__(self, name: str, positions: np.ndarray, attributes: np.ndarray, tensor_smith: "TensorSmith" = None):
        """Lidar points

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        positions : np.ndarray
            of shape (N, 3), usually in ego-system
        attributes : np.ndarray
            of shape (N, x)
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.positions = positions
        self.attributes = attributes
        self.tensor_smith = tensor_smith

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


class Bbox3D(SpatialTransformable):
    def __init__(self, name: str, elements: List[dict], dictionary: dict, flip_aware_class_pairs: List[tuple] = [], tensor_smith: "TensorSmith" = None):
        """
        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        elements : List[dict]
            a list of boxes. Each element is a dict of box having the following format:
            elements[0] = {
                'class': 'class.vehicle.passenger_car',
                'attr': [
                    'attr.time_varying.object.state.stationary',
                    'attr.vehicle.is_trunk_open.false',
                    'attr.vehicle.is_door_open.false'
                ],
                'size': [4.6486, 1.9505, 1.5845],
                'rotation': array([[ 0.93915682, -0.32818596, -0.10138267],
                                [ 0.32677338,  0.94460343, -0.03071667],
                                [ 0.1058472 , -0.00428138,  0.99437319]]),
                'translation': array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # NOTE: it is a column vector
                'track_id': '10035_0', # NOT USED
                'velocity': array([[0.], [0.], [0.]]) # NOTE: it is a column vector
            }
        dictionary : dict
            dictionary = {
                'classes': ['car', 'bus', 'pedestrain', ...],
                'attrs': []
            }
        flip_aware_class_pairs : List[tuple]
            list of class pairs that are flip-aware
            flip_aware_class_pairs = [('left_arrow', 'right_arrow')]
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        assert dictionary is not None
        self.elements = elements
        self.dictionary = dictionary
        self.remove_elements_not_recognized_by_dictionary()
        self.flip_aware_class_pairs = flip_aware_class_pairs
        self.tensor_smith = tensor_smith

    def remove_elements_not_recognized_by_dictionary(self):
        full_set_of_classes = {c for c in self.dictionary['classes']}
        for i in range(len(self.elements) - 1, -1, -1):
            if self.elements[i]['class'] not in full_set_of_classes:
                del self.elements[i]
    
    @property
    def corners(self) -> np.ndarray:
        """Convert Bboxed.elements to corners representation using copious

        Returns
        -------
        np.ndarray
            the output is of shape [len(self.elements), 8, 3] 
            The order of corner points look as follows

                (2) +---------+. (3)
                    | ` .   fr|  ` .
                    | (6) +---+-----+ (7)
                    |     |   |   bk|
                (1) +-----+---+. (0)|
                    ` .   |     ` . |
                    (5) ` +---------+ (4)
        """
        corners = []
        for ele in self.elements:
            copious_box3d = CopiousBox3d(
                position=ele['translation'].flatten(), 
                scale=np.array(ele['size']), 
                rotation=Rotation.from_matrix(ele['rotation'])
            )
            corners.append(copious_box3d.corners)
        return np.array(corners)

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        
        # in the mirror world, assume that a object is left-right symmetrical
        # however, y-axis of object coordinate is left-right
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        for ele in self.elements:
            ele['rotation'] = flip_mat @ ele['rotation'] @ flip_mat_self.T
            # here translation is a column array
            ele['translation'] = flip_mat @ ele['translation']
            ele['velocity'] = flip_mat @ ele['velocity']
        
        # flip classname for arrows
        for pair in self.flip_aware_class_pairs:
            for ele in self.elements:
                if ele['class'] in pair:
                    ele['class'] = pair[::-1][pair.index(ele['class'])]
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        # P_e' = R_e'e @ P_e
        for ele in self.elements:
            ele['rotation'] = rmat @ ele['rotation']
            ele['translation'] = rmat @ ele['translation']
            ele['velocity'] = rmat @ ele['velocity']
        return self


class Polyline3D(SpatialTransformable):
    def __init__(self, name: str, elements: List[dict], dictionary: dict, flip_aware_class_pairs: List[tuple] = [], tensor_smith: "TensorSmith" = None):
        """

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        elements : List[dict]
            a list of polylines. Each element is a dict of polyline having the following format:
            ```python
            elements[0] = {
                'class': 'class.road_marker.lane_line',
                'attr': <list>,
                'points': <N x 3 array>
            }
            ```
        dictionary : dict
            ```python
            dictionary = {
                'classes': ['car', 'bus', 'pedestrain', ...],
                'attrs': []
            }
            ```
        flip_aware_class_pairs : List[tuple], default []
            list of class pairs that are flip-aware
            ```flip_aware_class_pairs = [('left_arrow', 'right_arrow')]```
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.elements = elements
        self.dictionary = dictionary
        self.remove_elements_not_recognized_by_dictionary()
        self.flip_aware_class_pairs = flip_aware_class_pairs
        self.tensor_smith = tensor_smith

    def remove_elements_not_recognized_by_dictionary(self):
        full_set_of_classes = {c for c in self.dictionary['classes']}
        for i in range(len(self.elements) - 1, -1, -1):
            if self.elements[i]['class'] not in full_set_of_classes:
                del self.elements[i]

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # here points is a row array
        for ele in self.elements:
            ele['points'] = ele['points'] @ flip_mat.T
        
        # flip classname for arrows
        for pair in self.flip_aware_class_pairs:
            for ele in self.elements:
                if ele['class'] in pair:
                    ele['class'] = pair[::-1][pair.index(ele['class'])]
        
        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for ele in self.elements:
            ele['points'] = ele['points'] @ rmat.T
        return self


class Polygon3D(Polyline3D):
    pass


class ParkingSlot3D(SpatialTransformable):
    def __init__(self, name: str, elements: List[dict], dictionary: dict, tensor_smith: "TensorSmith" = None):
        """
        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        elements : List[dict]
            a list of polylines. Each element is a dict of polyline having the following format:
            ```python
            elements[0] = {
                'class': 'class.parking.parking_slot',
                'attr': <list>,
                'points': <4 x 3 array>
            }
            ```

        dictionary : dict
            ```python
            dictionary = {
                'classes': ['class.parking.parking_slot'],
                'attrs': []
            }

        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.elements = elements
        self.dictionary = dictionary
        self.tensor_smith = tensor_smith
        self.remove_elements_not_recognized_by_dictionary()
    
    def remove_elements_not_recognized_by_dictionary(self):
        full_set_of_classes = {c for c in self.dictionary['classes']}
        for i in range(len(self.elements) - 1, -1, -1):
            if self.elements[i]['class'] not in full_set_of_classes:
                del self.elements[i]
    

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        
        # here points is a row array
        for parkslot in self.elements:
            parkslot['points'] = parkslot['points'] @ flip_mat.T
        
            # in the mirror world, the assumed order of parking slot corners is broken, 
            # so we need to manually change the order
            parkslot['points'] = parkslot['points'][[1, 0, 3, 2], :]

        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for ele in self.elements:
            ele['points'] = ele['points'] @ rmat.T
        return self


class EgoPose(SpatialTransformable):
    def __init__(self, name: str, timestamp: str, rotation: np.ndarray, translation: np.ndarray, tensor_smith: "TensorSmith" = None):
        """The pose in 3D space of a given timestamp. 

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        timestamp : str
            corresponding timestamp of the pose
        rotation : np.ndarray
            rotation matrix, of shape (3, 3)
        translation : np.ndarray
            translation vector, of shape (1, 3), column vector
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.timestamp = timestamp
        self.rotation = rotation
        self.translation = translation
        self.tensor_smith = tensor_smith

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        
        # in the mirror world, assume that a object is left-right symmetrical
        # however, y-axis of object coordinate is left-right
        flip_mat_self = np.array([[1,  0, 0],
                                  [0, -1, 0], # apply了flip_map之后，坐标系变为了左手坐标系。用flip_mat_self将其重新转回右手坐标系
                                  [0,  0, 1]], dtype=np.float64)
        self.rotation = flip_mat @ self.rotation @ flip_mat_self.T
        self.translation = flip_mat @ self.translation  # self.translation is a column vector

        return self
    
    def rotate_3d(self, rmat, **kwargs):
        self.rotation = self.rotation @ rmat.T

        return self
    
    @property
    def trans_mat(self) -> np.array:
        _trans_mat = np.eye(4)
        _trans_mat[:3, :3] = self.rotation
        _trans_mat[:3, 3] = self.translation.flatten()
        return _trans_mat


class EgoPoseSet(TransformableSet):
    transformable_cls = EgoPose


class SegBev(SpatialTransformable):
    def flip_3d(self, *args, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, *args, **kwargs):
        raise NotImplementedError


class OccSdfBev(SpatialTransformable):
    def __init__(
        self,
        name: str, 
        src_voxel_range: tuple,
        occ: np.ndarray,
        height: np.ndarray,
        sdf: np.ndarray = None,
        mask: np.ndarray = None,
        tensor_smith: "TensorSmith" = None
    ):
        """OccSdfBev is a transformable contains occ, sdf and ground height info in a BEV view (a 2D spatial view).

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        src_voxel_range : tuple
            voxel_range=([-0.5, 2.5], [-15, 15], [15, -15]), from axis min to max
        occ : np.ndarray
            occ info, of shape (X, Y, C), where C denote the nubmer of occ classes
        sdf : np.ndarray
            sdf info, of shape (X, Y)
        height : np.ndarray
            height of the ground, of shape (X, Y)
        mask : np.ndarray, optional
            if provided, only positions with value 1 will be take into consideration, by default None
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.src_voxel_range = src_voxel_range
        self.occ = occ
        self.height = height
        self.tensor_smith = tensor_smith
        self._bev_shape = self.occ.shape[:2]
        self._bev_intrinsics = self._calc_bev_intrinsics()
        self.sdf = self._generate_sdf() if sdf is None else sdf
        self.mask = mask if mask is not None else np.ones_like(self.sdf, dtype=np.uint8)
        self._src_ego_points = self._unproject_bev_to_ego()
        # src ego coords is different from ego coords
        _, _, fx, fy = self._bev_intrinsics
        self._es_mat = np.float32([
            [-np.sign(fx), 0, 0], 
            [0, -np.sign(fy), 0], 
            [0, 0, 1]
        ])

    def _generate_sdf(self):
        cx, cy, fx, fy = self._bev_intrinsics
        df_obstacle = distance_transform_edt(1 - self.occ[..., 1] / 255)
        df_freespace = distance_transform_edt(self.occ[..., 1] / 255)
        assert abs(abs(fx) - abs(fy)) < 1e-3, f'fx={fx:.2f}, fy={fy:.2f}'
        sdf = (df_freespace - df_obstacle) / abs(fx)
        return sdf

    def _calc_bev_intrinsics(self):
        X, Y = self._bev_shape
        fx = X / (self.src_voxel_range[1][1] - self.src_voxel_range[1][0])
        fy = Y / (self.src_voxel_range[2][1] - self.src_voxel_range[2][0])
        cx = - self.src_voxel_range[1][0] * fx - 0.5
        cy = - self.src_voxel_range[2][0] * fy - 0.5

        return cx, cy, fx, fy

    def _unproject_bev_to_ego(self):
        cx, cy, fx, fy = self._bev_intrinsics
        X, Y = self._bev_shape
        vv, uu = np.meshgrid(np.arange(X), np.arange(Y), indexing='ij')
        xx = (vv - cx) / fx
        yy = (uu - cy) / fy
        zz = self.height
        # column points
        return np.stack([xx, yy, zz], axis=0).reshape(3, -1)
    

    def _project_ego_to_bev(self, ego_points):
        xx, yy, _ = ego_points
        cx, cy, fx, fy = self._bev_intrinsics
        vv_ = xx * fx + cx
        uu_ = yy * fy + cy
        return uu_.astype(np.float32), vv_.astype(np.float32)


    def flip_3d(self, flip_mat, **kwargs):
        # 1. get ego coordinates from bev
        # 2. apply flip_mat
        # 3. project to bev
        # 4. remap
        # flip_mat = M_e'e = M_ee'.T
        # P_e' = M_e'e @ P_e = flip_mat @ P_e
        assert flip_mat[2, 2] == 1, 'up-down flipping is unnecessary!'
        flipped_points = flip_mat @ self._src_ego_points
        uu_, vv_ = self._project_ego_to_bev(flipped_points)
        
        self.occ = cv2.remap(
            self.occ, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_NEAREST
        )
        self.sdf = cv2.remap(
            self.sdf, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_LINEAR
        )
        self.height = cv2.remap(
            self.height, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_LINEAR
        )
        self.mask = cv2.remap(
            self.mask, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_NEAREST
        )
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        # 1. get ego coordinates from bev
        # 2. apply rotation
        # 3. project to bev

        # rmat = R_e'e = R_ee'.T
        # P_s' = M_s'e' @ R_e'e @ M_es @ P_s = M_s'e' @ rmat @ M_es @ P_s
        
        rotated_points = self._es_mat.T @ rmat @ self._es_mat @ self._src_ego_points
        uu_, vv_ = self._project_ego_to_bev(rotated_points)
        
        self.occ = cv2.remap(
            self.occ, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_NEAREST
        )
        self.sdf = cv2.remap(
            self.sdf, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_LINEAR
        )
        self.height = cv2.remap(
            self.height, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_LINEAR
        )
        self.mask = cv2.remap(
            self.mask, 
            uu_.reshape(self._bev_shape),
            vv_.reshape(self._bev_shape),
            interpolation=cv2.INTER_NEAREST
        )
        return self


class OccSdf3D(SpatialTransformable):
    def flip_3d(self, *args, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, *args, **kwargs):
        raise NotImplementedError


class Variable(Transformable):
    def __init__(self, name: str, value: Any, tensor_smith: "TensorSmith" = None):
        """Arbitrary Variable (usually, we use this transformable to pass static data through)

        Parameters
        ----------
        name : str
            arbitrary string, will be set to each Transformable object to distinguish it with others
        value : Any
            arbitrary variable value that needs to pass through
        tensor_smith : TensorSmith, optional
            a tensor smith object, providing ToTensor for the transformable, by default None
        """
        super().__init__(name)
        self.value = deepcopy(value)
        self.tensor_smith = tensor_smith

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}, {self.value})'

#--------------------------------#




class Transform:
    '''
    Basic class for Transform.
    '''
    def __init__(self, *, scope="frame", **kwargs):
        assert scope.lower() in ["frame", "batch", "group", "transformable"]
        self.scope = scope.lower()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RandomTransform(Transform, metaclass=abc.ABCMeta):
    def __call__(self, *transformables, seeds=None, **kwargs):
        if seeds: # IMPORTANT: To make sure every thing is synchonized when scope in ['batch', 'group']
            _scope = "frame" if self.scope == 'transformable' else self.scope
            random.seed(seeds[_scope])
        if random.random() > self.prob:
            return list(transformables)
        return self._apply(*transformables, seeds=seeds, **kwargs)
    
    @abc.abstractmethod
    def _apply(self, *transformables, seeds=None, **kwargs):
        pass


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
        RandomTransform.__init__(self, scope=scope)
        self.prob = prob
        self.param_randomization_rules = param_randomization_rules or {}
        self.kwargs = kwargs
        self.validate_param_randomization_rules()

    def validate_param_randomization_rules(self):
        assert isinstance(
            self.param_randomization_rules, dict
        ), f"param_randomization_rules should be a dict. But {self.param_randomization_rules} is given."
        valid_rule_types = [ "float", "int", "enum", ]
        for _, rule in self.param_randomization_rules.items():
            assert rule["type"] in valid_rule_types, f"Only 'float', 'int' and 'enum' are valid types for a rule. But {rule['type']} is given."
            if rule["type"] == "enum":
                assert "choices" in rule, "choices should be used along with type: enum."
            else:
                assert "range" in rule, "range should be used along with type: float or int."

    def random_pick_param(self, rule: dict, seeds):
        random.seed(seeds[self.scope])  # why set seed inside? we want the same random results no matter what order of params it is.
        if rule["type"] == "float":
            return random.uniform(*rule["range"])
        elif rule["type"] == "int":
            return random.randint(*rule["range"])
        elif rule["type"] == "enum":
            return random.choice(rule["choices"])
        else:
            raise NotImplementedError

    def _apply(self, *transformables, seeds=None, **kwargs):
        for i, transformable in enumerate(transformables):
            _seed = dict(**seeds, transformable=make_seed(seeds["frame"], i))
            params = {param_name: self.random_pick_param(rule, _seed)
                      for param_name, rule in self.param_randomization_rules.items()}
            getattr(transformable, transform_func)(**params)
        return transformables

    return type(
        cls_name,
        (RandomTransform,),
        {
            "__init__": __init__,
            "_apply": _apply,
            "validate_param_randomization_rules": validate_param_randomization_rules,
            "random_pick_param": random_pick_param,
        },
    )


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

ToTensor = deterministic_transform_class_factory("ToTensor", "to_tensor")

#######################
# Customed Transforms #
#######################
class RandomSetIntrinsicParam(RandomTransform):

    def __init__(self, *, prob: float = 0.2, jitter_ratio: float = 0.01, scope: str = "transformable", **kwargs):
        """This transform random jitters the intrinsic params of CameraTransformable.
        (NOTE: only applicable to cx, cy, fx, fy).

        Parameters
        ----------
        prob : float, optional
            the happening probability of this Transform, value range [0, 1], by default 0.2
        jitter_ratio : float, optional
            the relative ratio of the jittering to each param in intrinsic, by default 0.01 (i.e. 0%)
        scope : str, optional
            the scope of the Transform, by default "transformable"
        """
        Transform.__init__(self, scope=scope)
        self.prob = prob
        self.jitter_ratio = jitter_ratio
        self.kwargs = kwargs

    def random_pick_param(self, seeds, base_value):
        random.seed(seeds[self.scope])
        return random.uniform(1 - self.jitter_ratio, 1 + self.jitter_ratio) * base_value

    def _apply(self, *transformables, seeds=None, **kwargs):
        for i, transformable in enumerate(transformables):
            if isinstance(transformable, CameraTransformable):
                _seeds = dict(**seeds, transformable=make_seed(seeds["frame"], i))
                new_intrinsic = [self.random_pick_param(_seeds, param) for param in transformable.intrinsic]
                transformable.set_intrinsic_param(new_intrinsic)
            elif isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for t in transformable.transformables.values():
                    _seeds = dict(**seeds, transformable=make_seed(seeds["frame"], i))
                    new_intrinsic = [self.random_pick_param(_seeds, param) for param in t.intrinsic]
                    t.set_intrinsic_param(new_intrinsic)
        return transformables


class RandomSetExtrinsicParam(RandomTransform):
    def __init__(self, *, prob: float = 0.2, angle: float = 1.0, translation: float = 0.01, scope: str = "transformable", **kwargs):
        """This transform random jitters the rotation angle and translation of CameraTransformable's extrinsic params.

        Parameters
        ----------
        prob : float, optional
            the happening probability of this Transform, value range [0, 1], by default 0.2
        angle : float, optional
            the jittering angle (degrees) that applies to current extrinsic.Rotation, by default 1.0
        translation : float, optional
            the translation (m) that applies to current extrinsic.Translation, by default 0.01
        scope : str, optional
            the scope of the Transform, by default "transformable"
        """
        Transform.__init__(self, scope=scope)
        self.prob = prob
        self.angle = angle
        self.translation = translation
        self.kwargs = kwargs

    def random_pick_param(self, seeds):
        random.seed(seeds[self.scope])

        def _get_random_value(deviation):
            deviation_abs = abs(deviation)
            return random.uniform(-deviation_abs, deviation_abs)

        def _get_random_delta_rotation():
            return Rotation.from_euler(
                'xyz', 
                [_get_random_value(self.angle) for _ in range(3)],
                degrees=True
            ).as_matrix()
        
        def _get_random_delta_translation():
            return np.array([_get_random_value(self.translation) for _ in range(3)])
        
        return _get_random_delta_rotation(),  _get_random_delta_translation()

    def _apply(self, *transformables, seeds=None, **kwargs):
        for i, transformable in enumerate(transformables):
            if isinstance(transformable, CameraTransformable):
                _seeds = dict(**seeds, transformable=make_seed(seeds['frame'], i))
                delta_R, delta_t = self.random_pick_param(_seeds)
                R, t = transformable.extrinsic
                new_R = delta_R @ R
                new_t = delta_t + t
                transformable.set_extrinsic_param([new_R, new_t])
            elif isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for t_sub in transformable.transformables.values():
                    _seeds = dict(**seeds, transformable=make_seed(seeds['frame'], i))
                    delta_R, delta_t = self.random_pick_param(_seeds)
                    R, t = t_sub.extrinsic
                    new_R = delta_R @ R
                    new_t = delta_t + t
                    t_sub.set_extrinsic_param([new_R, new_t])
        return transformables


class RandomChooseKTransform(RandomTransform):
    def __init__(self, transforms, *, prob=0.5, K=1, transform_probs=None, scope="frame"):
        super().__init__(scope=scope)
        self.transforms = transforms
        if transform_probs is not None:
            assert len(transforms) == len(transform_probs)
        self.transform_probs = transform_probs
        self.prob = prob
        self.K = K
    
    def _apply(self, *transformables, seeds=None, **kwargs):
        transforms = np.random.choice(
            self.transforms,
            size=self.K,
            replace=False,
            p=self.transform_probs,
        )
        for transform in transforms:
            transform(*transformables, **kwargs, seeds=seeds)
        return transformables


class RandomImageISP(Transform):
    def __init__(self, prob=0.5, transform_probs=None, scope='transformable'):
        super().__init__(scope=scope)
        self.transforms=[
            RandomBrightness(prob=0.5, param_randomization_rules={"brightness": {"type": "float", "range": [0.5, 2.0]}}),
            RandomSaturation(prob=0.5, param_randomization_rules={"saturation": {"type": "float", "range": [0.0, 2.0]}}),
            RandomContrast(prob=0.5, param_randomization_rules={"contrast": {"type": "float", "range": [0.5, 2.0]}}),
            RandomHue(prob=0.5, param_randomization_rules={"hue": {"type": "float", "range": [-0.5, 0.5]}}),
            RandomSharpness(prob=0.5, param_randomization_rules={"sharpness": {"type": "float", "range": [0.0, 2.0]}}),
            RandomPosterize(prob=0.5, param_randomization_rules={"bits": {"type": "int", "range": [4, 8]}}),
            RandomChannelShuffle(prob=0.5),
            RandomAutoContrast(prob=0.2),
            RandomSolarize(prob=0.01),
            RandomImEqualize(prob=0.1),
        ]
        for transform in self.transforms:
            transform.scope = scope
        self.delegate_transform = RandomChooseKTransform(
            transforms=self.transforms,
            prob=prob,
            K=len(self.transforms),
            transform_probs=transform_probs,
        )
    
    def __call__(self, *transformables, seeds=None, **kwargs):
        return self.delegate_transform(*transformables, seeds=seeds)


class RandomImageTransformAT(Transform):
    pass



class RandomImageOmit(Transform):
    pass


class BGR2RGB(Transform):
    def __init__(self, scope="transformable"):
        super().__init__(scope=scope)
        self.order_rgb = [2, 1, 0]

    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, CameraImageSet):
                for t in transformable.transformables.values():
                    t.channel_shuffle(order=self.order_rgb)
            elif isinstance(transformable, CameraImage):
                transformable.channel_shuffle(order=self.order_rgb)
        return transformables


class RenderIntrinsic(Transform):
    
    def __init__(self, resolutions, intrinsics='default', scope="frame"):
        '''
        resolutions: {<cam_id>: (W, H), ...}
        intrinsics: {<cam_id>: (cx, cy, fx, fy, ...), ...}
        '''
        super().__init__(scope=scope)
        self.cam_ids = list(resolutions.keys())
        self.resolutions = resolutions
        if intrinsics == 'default':
            intrinsics = {}
            for cam_id in self.cam_ids:
                intrinsics[cam_id] = 'default'
        # fill default intrinsics
        for cam_id in self.cam_ids:
            if cam_id not in intrinsics:
                intrinsics[cam_id] = 'default'
        self.intrinsics = intrinsics

    @staticmethod
    def _get_default_intrinsic(resolution, cam_type):
        W, H = resolution
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        if cam_type in ['FisheyeCamera']:
            fx = fy = W / 4
        else:
            fx = fy = W / 2
        intrinsic = [cx, cy, fx, fy]
        return intrinsic
    
    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for t in transformable.transformables.values():
                    cam_id = t.cam_id
                    if cam_id in self.cam_ids:
                        resolution = self.resolutions[cam_id]
                        intrinsic = self.intrinsics[cam_id]
                        if intrinsic == 'default':
                            intrinsic = self._get_default_intrinsic(resolution, t.cam_type)
                        t.render_intrinsic(resolution, intrinsic)
            elif isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                cam_id = transformable.cam_id
                if cam_id in self.cam_ids:
                    resolution = self.resolutions[cam_id]
                    intrinsic = self.intrinsics[cam_id]
                    if intrinsic == 'default':
                        intrinsic = self._get_default_intrinsic(resolution, transformable.cam_type)
                    transformable.render_intrinsic(resolution, intrinsic)
        return transformables


class RenderExtrinsic(Transform):
    
    def __init__(self, del_rotations: dict, scope="frame"):
        """_summary_

        Parameters
        ----------
        del_rotations : dict
            ```python
            {<cam_id>: [<ang>, <ang>, <ang>], # ego x-y-z euler angles
            ...}
            ```
        scope : str, optional
            seed scope, by default "frame"
        """
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
            if isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for t in transformable.transformables.values():
                    cam_id = t.cam_id
                    if cam_id in self.cam_ids:
                        t.render_extrinsic(self.del_extrinsics[cam_id])
            elif isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                cam_id = transformable.cam_id
                if cam_id in self.cam_ids:
                    transformable.render_extrinsic(self.del_extrinsics[cam_id])
        return transformables


class RenderVirtualCamera(Transform):
    
    def __init__(self, camera_settings: dict, scope="frame"):
        """
        Notes
        -----
        The extrinsic parameters must be in original ego system, 
        in which the camera is calibrated.

        Parameters
        ----------
        camera_settings : dict
            ```python
            {<cam_id>: dict(cam_type='PerspectiveCamera' | 'FisheyeCamera',
                            resolution=(W, H),
                            euler_angles=[<ang>, <ang>, <ang>], # ego x-y-z euler angles,
                            intrinsic='auto' or (cx, cy, fx, fy, *dist_params)),
            ...}
            ```
        scope : str, optional
            seed scope, by default "frame"
        """
        super().__init__(scope=scope)
        self.cam_ids = list(camera_settings.keys())
        for cam_id in self.cam_ids:
            assert camera_settings[cam_id]['cam_type'] in ['PerspectiveCamera', 'FisheyeCamera']
        self.cameras = {}
        for cam_id in self.cam_ids:
            resolution = camera_settings[cam_id]['resolution']
            extrinsic = (
                Rotation.from_euler('xyz', camera_settings[cam_id]['euler_angles'], degrees=True).as_matrix(),
                np.array((0, 0, 0))
            )
            intrinsic = camera_settings[cam_id]['intrinsic']
            cam_type = camera_settings[cam_id]['cam_type']
            if intrinsic == 'auto': 
                W, H = resolution
                if cam_type == 'PerspectiveCamera':
                    cx = (W - 1) / 2
                    cy = (H - 1) / 2
                    fx = fy = W / 2
                    intrinsic = [cx, cy, fx, fy]
                if cam_type == 'FisheyeCamera':
                    cx = (W - 1) / 2
                    cy = (H - 1) / 2
                    fx = fy = W / 4
                    intrinsic = [cx, cy, fx, fy, 0.1, 0, 0, 0]
            camera_class = getattr(vc, camera_settings[cam_id]['cam_type'])
            self.cameras[cam_id] = camera_class(resolution, extrinsic, intrinsic)            
    
    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for t_sub in transformable.transformables.values():
                    cam_id = t_sub.cam_id
                    if cam_id in self.cam_ids:
                        t_sub.render_camera(self.cameras[cam_id])
            elif isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                cam_id = transformable.cam_id
                if cam_id in self.cam_ids:
                    transformable.render_camera(self.cameras[cam_id])
        return transformables




class RandomRenderExtrinsic(RandomTransform):
    def __init__(self, *, prob=0.5, angles=[1, 1, 1], scope="frame", **kwargs):
        super().__init__(scope=scope)
        self.prob = prob
        self.angles = angles

    def random_pick_param(self, seeds):
        random.seed(seeds[self.scope])

        del_R = Rotation.from_euler(
            'xyz', 
            [random.uniform(-self.angles[0], self.angles[0]),
             random.uniform(-self.angles[1], self.angles[1]),
             random.uniform(-self.angles[2], self.angles[2])],
            degrees=True
        ).as_matrix()
        return (del_R, np.array([0, 0, 0]))

    def _apply(self, *transformables, seeds=None, **kwargs):
        for i, transformable in enumerate(transformables):
            if isinstance(transformable, (CameraImageSet, CameraSegMaskSet, CameraDepthSet)):
                for j, t in enumerate(transformable.transformables.values()):
                    _seeds = dict(**seeds, transformable=make_seed(seeds['frame'], i, j))
                    params = self.random_pick_param(_seeds)
                    t.render_extrinsic(params)

            elif isinstance(transformable, (CameraImage, CameraSegMask, CameraDepth)):
                _seeds = dict(**seeds, transformable=make_seed(seeds['frame'], i))
                params = self.random_pick_param(_seeds)
                transformable.render_extrinsic(params)


class RandomRotateSpace(RandomTransform):
    def __init__(self, *, prob=0.5, angles=[2, 2, 10], 
                 prob_inverse_cameras_rotation=0.5, scope="group", **kwargs):
        '''
        angles = [roll, pitch, yaw] in degrees, x-y-z
        '''
        super().__init__(scope=scope)
        self.prob = prob
        self.angles = angles
        self.prob_inverse_cameras_rotation = prob_inverse_cameras_rotation

    def random_pick_param(self, seeds):
        random.seed(seeds[self.scope])
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
        
        return del_R, inverse_cameras_rotation

    def _apply(self, *transformables, seeds=None, **kwargs):
        for i, transformable in enumerate(transformables):
            if transformable is not None:
                _seeds = dict(**seeds, transformable=make_seed(seeds["frame"], i))
                del_R, inverse_cameras_rotation = self.random_pick_param(_seeds)
                transformable.rotate_3d(del_R)
                if inverse_cameras_rotation:
                    del_extrinsic = (del_R.T, np.array([0.0, 0, 0]))
                    transformable.render_extrinsic(del_extrinsic)
        
        return transformables


class RandomMirrorSpace(RandomTransform):
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

    def _apply(self, *transformables, seeds=None, **kwargs):
        for transformable in transformables:
            if transformable is not None:
                transformable.flip_3d(self.flip_mat)
        
        return transformables


class MirrorTime(Transform):
    pass



class ScaleTime(Transform):
    pass


# TODO: Implement this function for the model that relies on or need to predict objects' trajectory.
def extract_trajectory_from_bbox3d(bbox3d_list: List[Bbox3D], ego_poses: EgoPoseSet) -> List[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    """Extract trajectory of each distinct obj_track_id from a list of Bbox3D. 

    Parameters
    ----------
    bbox3d_list : List[Bbox3D]
        a list of input Bbox3D in current group
    ego_poses: EgoPoseSet
        the ego-poses of current group
    Returns
    -------
    List[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]
        Trajectories of each object (i.e. obj_track_id) in each frame of the group. 
        The return is of the same length as `bbox3d_list` and `ego_poses`.
        Say we have 3 frames in current group, below is an examplar return: 
            [
                {
                    "10000_0": [(R0, t0), (R1, t1), (R2, t2)],
                    "10001_0": [(R0, t0), (R1, t1)],
                },
                {
                    "10000_0": [(R0, t0), (R1, t1), (R2, t2)],
                    "10001_0": [(R0, t0), (R1, t1)],
                    "10002_0": [(R1, t1), (R2, t2)],
                },
                {
                    "10000_0": [(R0, t0), (R1, t1), (R2, t2)],
                    "10002_0": [(R1, t1), (R2, t2)],
                },
            ]
    """
    # Some initial thoughts and notes:
    # 1. we have to traverse through all the bbox3d in `bbox3d_list` to created object_track_id centric representation
    # 2. determine which trajectories to expose for each frame based on what object it has.
    # 3. all poses of the same obj_track_id should be converted to current frame's ego-coordsys.
    # 4. any one who want to use this function should overwrite __call__ in the model feeder class, because it needs cross-time info
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
    BGR2RGB,
    RenderIntrinsic, 
    RenderExtrinsic, 
    RenderVirtualCamera, 
    RandomRenderExtrinsic, 
    RandomRotateSpace, 
    RandomMirrorSpace, 
    RandomSetIntrinsicParam, 
    RandomSetExtrinsicParam, 
]

available_transformables = [
    CameraImage, CameraImageSet, CameraSegMask, CameraSegMaskSet,
    CameraDepth, CameraDepthSet, LidarPoints, Bbox3D,
    Polyline3D, Polygon3D, ParkingSlot3D, EgoPose,
    EgoPoseSet, SegBev, OccSdfBev, OccSdf3D,
]

for transform in available_transforms:
    TRANSFORMS.register_module(module=transform)

for transformable in available_transformables:
    TRANSFORMABLES.register_module(module=transformable)


__all__ = [t.__name__ for t in available_transforms] + [t.__name__ for t in available_transformables]
