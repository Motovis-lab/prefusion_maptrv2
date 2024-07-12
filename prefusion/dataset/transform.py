import random
import inspect

import cv2
import mmcv
import numpy as np

import virtual_camera as vc
# import albumentations as AT

from scipy.spatial.transform import Rotation

from prefusion.registry import TRANSFORMS


class Transformable:
    """
    Base class for all transformables.
    """

    def __init__(self, data: dict):
        self.data = data
    
    def __repr__(self):
        return "type: {}\ndata: {}".format(self.__class__.__name__, self.data)
    

    def at_transform(self, func_name, **kwargs):
        return self

    def adjust_brightness(self, brightness=1, **kwargs):
        return self
    
    def adjust_saturation(self, saturation=1, **kwargs):
        return self
    
    def adjust_contrast(self, contrast=1, **kwargs):
        return self
    
    def adjust_hue(self, hue=1, **kwargs):
        return self
    
    def adjust_sharpness(self, sharpness=1, **kwargs):
        return self
    
    def posterize(self, bits=8, **kwargs):
        return self
    
    def auto_contrast(self, **kwargs):
        return self
    
    def imequalize(self, **kwargs):
        return self

    def solarize(self, **kwargs):
        return self

    def sobelize(self, **kwargs):
        return self

    def gaussian_blur(self, **kwargs):
        # use albumentations
        return self

    def channel_shuffle(self, **kwargs):
        return self
    
    def intrinsic_jitter(self, **kwargs):
        raise self
    
    def apply_intrinsic(self, **kwargs):
        raise NotImplementedError
    
    def extrinsic_jitter(self, **kwargs):
        raise self
    
    def apply_extrinsic(self, **kwargs):
        raise NotImplementedError

    def flip_3d(self, **kwargs):
        raise NotImplementedError

    def scale_3d(self, **kwargs):
        raise NotImplementedError
    
    def rotate_3d(self, **kwargs):
        raise NotImplementedError

    def to_tensor(self, **kwargs):
        return self

    def get_data(self, **kwargs):
        return self.data



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



class Image(Transformable):
    """
    - self.data = {
        'img': <img_arr>,
        'ego_mask': <arr>,
        'cam_id': <str>,
        'cam_type': < 'FisheyeCamera' | 'PerspectiveCamera' >
        'extrinsic': (R, t),
        'intrinsic': [cx, cy, fx, fy, *distortion_params]
    }
    """

    def __init__(self, data: dict):
        super().__init__(data)
        assert self.data['cam_type'] in ['FisheyeCamera', 'PerspectiveCamera']

    def at_transform(self, func_name, **kwargs):
        func = getattr(AT, func_name)(**kwargs)
        self.data['img'] = func(image=self.data['img'])['image']
        return self

    def adjust_brightness(self, brightness=1, **kwargs):
        self.data['img'] = mmcv.adjust_brightness(self.data['img'], factor=brightness)
        return self
    
    def adjust_saturation(self, saturation=1, **kwargs):
        self.data['img'] = mmcv.adjust_color(self.data['img'], alpha=saturation)
        return self
    
    def adjust_contrast(self, contrast=1, **kwargs):
        self.data['img'] = mmcv.adjust_contrast(self.data['img'], factor=contrast)
        return self
    
    def adjust_hue(self, hue=1, **kwargs):
        self.data['img'] = mmcv.adjust_hue(self.data['img'], hue_factor=hue)
        return self
    
    def adjust_sharpness(self, sharpness=1, **kwargs):
        self.data['img'] = mmcv.adjust_sharpness(self.data['img'], factor=sharpness)
        return self
    
    def posterize(self, bits=8, **kwargs):
        self.data['img'] = mmcv.posterize(self.data['img'], bits)
        return self
    
    def auto_contrast(self, **kwargs):
        self.data['img'] = mmcv.auto_contrast(self.data['img'])
        return self
    
    def imequalize(self, **kwargs):
        self.data['img'] = mmcv.imequalize(self.data['img'])
        return self
    
    def solarize(self, **kwargs):
        self.data['img'] = mmcv.solarize(self.data['img'])
        return self
    
    def channel_shuffle(self, order=[0, 1, 2], **kwargs):
        assert len(order) == self.data['img'].shape[2]
        self.data['img'] = self.data['img'][..., order]
        return self
    

    def intrinsic_jitter(self, percentile=0.5, **kwargs):
        cx, cy, fx, fy, *distortion_params = self.data['intrinsic']
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.data['intrinsic'] = [cx_, cy_, fx_, fy_, *distortion_params]
        return self
    

    def extrinsic_jitter(self, angle=1, translation=0.05, **kwargs):
        R, t = self.data['extrinsic']
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
        self.data['extrinsic'] = (R_, t_)
        return self
    

    def apply_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.data['intrinsic']), 'invalid intrinsic params'
        resolution_old = self.data['img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        camera_old = camera_class(
            resolution_old,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        if len(intrinsic) < len(self.data['intrinsic']):
            intrinsic_new = list(intrinsic) + self.data['intrinsic'][len(intrinsic):]
        else:
            intrinsic_new = intrinsic
        camera_new = camera_class(
            resolution,
            self.data['extrinsic'],
            intrinsic_new
        )
        self.data['img'], self.data['ego_mask'] = vc.render_image(self.data['img'], camera_old, camera_new)
        self.data['intrinsic'] = intrinsic_new
        
        return self
    
    def apply_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.data['img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        R, t = self.data['extrinsic']
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.data['intrinsic']
        )
        self.data['img'], self.data['ego_mask'] = vc.render_image(self.data['img'], camera_old, camera_new)
        self.data['extrinsic'] = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.data['extrinsic'][0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.data['extrinsic'][1] @ flip_mat.T
        self.data['extrinsic'] = (R_new, t_new)
        self.data['intrinsic'][0] = self.data['img'].shape[1] - 1 - self.data['intrinsic'][0]
        self.data['img'] = np.array(self.data['img'][:, ::-1])
        self.data['ego_mask'] = np.array(self.data['ego_mask'][:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.data['extrinsic']
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.data['extrinsic'] = (R_new, t_new)
        return self
    

    def to_tensor(self, **kwargs):
        return self



class LidarPoints(Transformable):
    '''
    self.data = {
        'positions': <N x 3 array>, in ego-systemssss
        'intensity': <N x 1 array>
    }
    '''

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # here points is a row array
        self.data['positions'] = self.data['positions'] @ flip_mat.T
        
        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        self.data['positions'] = self.data['positions'] @ rmat.T
        return self


    def to_tensor(self, **kwargs):
        return self



class ImageSegMask(Transformable):
    """
    - self.data = {
        'img': <seg_arr>,
        'ego_mask': <arr>,
        'cam_type': < 'FisheyeCamera' | 'PerspectiveCamera' >
        'extrinsic': (R, t),
        'intrinsic': [cx, cy, fx, fy, *distortion_params]
    }
    """

    def __init__(self, data: dict, dictionary: dict):
        super().__init__(data)
        assert self.data['cam_type'] in ['FisheyeCamera', 'PerspectiveCamera']


    def intrinsic_jitter(self, percentile=0.5, **kwargs):
        cx, cy, fx, fy, *distortion_params = self.data['intrinsic']
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.data['intrinsic'] = [cx_, cy_, fx_, fy_, *distortion_params]
        return self
    

    def extrinsic_jitter(self, angle=1, translation=0.05, **kwargs):
        R, t = self.data['extrinsic']
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
        self.data['extrinsic'] = (R_, t_)
        return self
    

    def apply_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.data['intrinsic']), 'invalid intrinsic params'
        resolution_old = self.data['img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        camera_old = camera_class(
            resolution_old,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        if len(intrinsic) < len(self.data['intrinsic']):
            intrinsic_new = list(intrinsic) + self.data['intrinsic'][len(intrinsic):]
        else:
            intrinsic_new = intrinsic
        camera_new = camera_class(
            resolution,
            self.data['extrinsic'],
            intrinsic_new
        )
        self.data['img'], self.data['ego_mask'] = vc.render_image(self.data['img'], camera_old, camera_new)
        self.data['intrinsic'] = intrinsic_new
        
        return self
    
    def apply_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.data['img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        R, t = self.data['extrinsic']
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.data['intrinsic']
        )
        self.data['img'], self.data['ego_mask'] = vc.render_image(
            self.data['img'], camera_old, camera_new, interpolation=cv2.INTER_NEAREST
        )
        self.data['extrinsic'] = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.data['extrinsic'][0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.data['extrinsic'][1] @ flip_mat.T
        self.data['extrinsic'] = (R_new, t_new)
        self.data['intrinsic'][0] = self.data['img'].shape[1] - 1 - self.data['intrinsic'][0]
        self.data['img'] = np.array(self.data['img'][:, ::-1])
        self.data['ego_mask'] = np.array(self.data['ego_mask'][:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.data['extrinsic']
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.data['extrinsic'] = (R_new, t_new)
        return self

    def to_tensor(self, **kwargs):
        return self



class ImageDepth(Transformable):
    """
    - self.data = {
        'dep_img': <depth_arr>,
        'ego_mask': <arr>,
        'cam_type': < 'FisheyeCamera' | 'PerspectiveCamera' >
        'extrinsic': (R, t),
        'intrinsic': [cx, cy, fx, fy, *distortion_params]
    }
    """
    def __init__(self, data: dict, depth_mode='d'):
        '''
        - depth_mode: 'z' or 'd'
          - 'z': depth in z axis of camera coordinate
          - 'd': depth in distance of point to camera optical point
        '''
        super().__init__(data)
        assert self.data['cam_type'] in ['FisheyeCamera', 'PerspectiveCamera']
        assert depth_mode in ['z', 'd']
        self.depth_mode = depth_mode


    def intrinsic_jitter(self, percentile=0.5, **kwargs):
        cx, cy, fx, fy, *distortion_params = self.data['intrinsic']
        scale = percentile * 0.01
        cx_ = random.uniform(1 - scale, 1 + scale) * cx
        cy_ = random.uniform(1 - scale, 1 + scale) * cy
        fx_ = random.uniform(1 - scale, 1 + scale) * fx
        fy_ = random.uniform(1 - scale, 1 + scale) * fy
        self.data['intrinsic'] = [cx_, cy_, fx_, fy_, *distortion_params]
        return self
    

    def extrinsic_jitter(self, angle=1, translation=0.05, **kwargs):
        R, t = self.data['extrinsic']
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
        self.data['extrinsic'] = (R_, t_)
        return self
    

    def apply_intrinsic(self, resolution, intrinsic, **kwargs):
        assert len(intrinsic) <= len(self.data['intrinsic']), 'invalid intrinsic params'
        resolution_old = self.data['dep_img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        camera_old = camera_class(
            resolution_old,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        if len(intrinsic) < len(self.data['intrinsic']):
            intrinsic_new = list(intrinsic) + self.data['intrinsic'][len(intrinsic):]
        else:
            intrinsic_new = intrinsic
        camera_new = camera_class(
            resolution,
            self.data['extrinsic'],
            intrinsic_new
        )
        self.data['dep_img'], self.data['ego_mask'] = vc.render_image(self.data['dep_img'], camera_old, camera_new)
        self.data['intrinsic'] = intrinsic_new
        
        return self
    
    def apply_extrinsic(self, delta_extrinsic, **kwargs):
        resolution = self.data['dep_img'].shape[:2][::-1]
        camera_class = getattr(vc, self.data['cam_type'])
        R, t = self.data['extrinsic']
        del_R, del_t = delta_extrinsic
        camera_old = camera_class(
            resolution,
            self.data['extrinsic'],
            self.data['intrinsic'],
            ego_mask=self.data['ego_mask']
        )
        R_new, t_new = del_R @ R, del_t + t
        camera_new = camera_class(
            resolution,
            (R_new, t_new),
            self.data['intrinsic']
        )
        # TODO: get real points from depth then remap to image
        if self.depth_mode == 'd':
            self.data['dep_img'], self.data['ego_mask'] = vc.render_image(
                self.data['dep_img'], camera_old, camera_new, interpolation=cv2.INTER_NEAREST
            )
        elif self.depth_mode == 'z':
            raise NotImplementedError
        self.data['extrinsic'] = (R_new, t_new)
        
        return self


    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        R_new = flip_mat @ self.data['extrinsic'][0] @ flip_mat_self.T
        # here translation is a row array
        t_new = self.data['extrinsic'][1] @ flip_mat.T
        self.data['extrinsic'] = (R_new, t_new)
        self.data['intrinsic'][0] = self.data['dep_img'].shape[1] - 1 - self.data['intrinsic'][0]
        self.data['dep_img'] = np.array(self.data['dep_img'][:, ::-1])
        self.data['ego_mask'] = np.array(self.data['ego_mask'][:, ::-1])
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        R, t = self.data['extrinsic']
        R_new = rmat @ R
        t_new = t @ rmat.T
        self.data['extrinsic'] = (R_new, t_new)
        return self

    def to_tensor(self, **kwargs):
        return self



class Bbox3D(Transformable):
    """
    - self.data = [element, element, ...]
    - element = {
        'class': 'class.vehicle.passenger_car',
        'attr': {'attr.time_varying.object.state': 'attr.time_varying.object.state.stationary',
                 'attr.vehicle.is_trunk_open': 'attr.vehicle.is_trunk_open.false',
                 'attr.vehicle.is_door_open': 'attr.vehicle.is_door_open.false'},
        'size': [4.6486, 1.9505, 1.5845],
        'rotation': array([[ 0.93915682, -0.32818596, -0.10138267],
                           [ 0.32677338,  0.94460343, -0.03071667],
                           [ 0.1058472 , -0.00428138,  0.99437319]]),
        'translation': array([[-15.70570354], [ 11.88484971], [ -0.61029085]]), # VERTICAL
        'track_id': '10035_0', # NOT USED
        'velocity': array([[0.], [0.], [0.]])
    }
    """

    def __init__(self, data: list, dictionary: dict):
        '''
        dictionary: branches of keys.
        dictionary = {
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

        '''
        self.dictionary = dictionary
        # filter elements by dictionary
        available_elements = []
        for branch in dictionary:
            available_elements.extend(branch['classes'])
        self.data = []
        for element in data:
            if element['class'] in available_elements:
                self.data.append(element)
        
    

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        for element in self.data:
            element['rotation'] = flip_mat @ element['rotation'] @ flip_mat_self.T
            # here translation is a row array
            element['translation'] = flip_mat @ element['translation']
            element['velocity'] = flip_mat @ element['velocity']
        
        return self
    

    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for element in self.data:
            element['rotation'] = rmat @ element['rotation']
            element['translation'] = rmat @ element['translation']
            element['velocity'] = rmat @ element['velocity']
        return self


    def to_tensor(self, **kwargs):
        return self
    


class BboxBev(Bbox3D):
    
    def to_tensor(self, **kwargs):
        return self



class Cylinder3D(Bbox3D):
    
    def to_tensor(self, **kwargs):
        return self



class OrientedCylinder3D(Bbox3D):
    
    def to_tensor(self, **kwargs):
        return self



class Square3D(Bbox3D):
    
    def to_tensor(self, **kwargs):
        return self



class Polyline3D(Transformable):
    '''
    - self.data = [element, element, ...]
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
            available_elements.extend(branch['classes'])
        self.data = []
        for element in data:
            if element['class'] in available_elements:
                self.data.append(element)

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # here points is a row array
        for element in self.data:
            element['points'] = element['points'] @ flip_mat.T
        
        return self
    
    def rotate_3d(self, rmat, **kwargs):
        # rmat = R_e'e = R_ee'.T
        # R_c = R_ec
        # R_c' = R_e'c = R_e'e @ R_ec
        for element in self.data:
            element['points'] = element['points'] @ rmat.T
        return self


    def to_tensor(self, bev_resolution, **kwargs):
        return self



class Polygon3D(Polyline3D):
    
    def to_tensor(self, **kwargs):
        return self



class ParkingSlot3D(Polyline3D):

    def flip_3d(self, flip_mat, **kwargs):
        assert flip_mat[2, 2] == 1, 'up down flip is unnecessary.'
        # in the mirror world, assume that a object is left-right symmetrical
        flip_mat_self = np.eye(3)
        flip_mat_self[1, 1] = -1
        # here points is a row array
        for element in self.data:
            element['points'] = flip_mat_self @ element['points'] @ flip_mat.T
        
        return self
    
    def to_tensor(self, **kwargs):
        return self



class Trajectory(Transformable):
    '''
    - self.data = [element, element, ...]
    - element = [(R, t), (R, t), ...]
    '''
    def __init__(self, data: list):
        self.data = data



class SegBev(Transformable):
    
    def to_tensor(self, **kwargs):
        return self



class OccSdfBev(Transformable):

    '''
    self.data = {
        'src_view_range': [back, front, right, left, bottom, up], # in ego system
        'dst_view_range': [back, front, right, left, bottom, up], # in ego system
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


    def to_tensor(self, dst_bev_resolution, **kwargs):
        return self



class OccSdf3D(Transformable):

    def to_tensor(self, **kwargs):
        return self

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
    def __init__(self, *, prob=0.0, scope="frame", **kwargs):
        Transform.__init__(self, scope=scope)
        self.prob = prob
        self.kwargs = kwargs

    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
        if random.random() > self.prob:
            return list(transformables)
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
RandomImequalize = random_transform_class_factory("RandomImEqualize", "imequalize")


RandomIntrinsicParam = random_transform_class_factory("RandomIntrinsicParam", "intrinsic_jitter")
RandomExtrinsicParam = random_transform_class_factory("RandomExtrinsicParam", "extrinsic_jitter")
ToTensor = deterministic_transform_class_factory("ToTensor", "to_tensor")



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

    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
        if random.random() <= self.prob:
            transform = random.choices(
                population=self.transforms,
                weights=self.transform_probs,
                k=1
            )[0]
            for transformable in transformables:
                transform(*transformable, **kwargs)
        return transformables



class MultiRandomTransforms(Transform):
    def __init__(self, transforms, *, scope='frame') -> None:
        self.transforms = transforms
        self.scope = scope
    
    def __repr__(self) -> str:
        for transform in self.transforms:
            print(transform.__name__)
        return 'A random sequence of transforms: '
    
    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if seed is not None:
                random.seed(seed)
        random.shuffle(self.transforms)
        for transform in self.transforms:
            for transformable in transformables:
                transform(*transformable, seeds=seeds, **kwargs)
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
    
    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
        
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



class IntrinsicImage(Transform):
    
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
                fx = W / 2
                fy = W / 2
                intrinsic = [cx, cy, fx, fy]
                self.intrinsics[cam_id] = intrinsic
        else:
            self.intrinsics = intrinsics
    
    def __call__(self, *transformables, **kwargs):
        for transformable in transformables:
            if isinstance(transformable, (Image, ImageSegMask, ImageDepth)):
                if transformable.data['cam_id'] in self.cam_ids:
                    resolution = self.resolutions[transformable.data['cam_id']]
                    intrinsic = self.intrinsics[transformable.data['cam_id']]
                    transformable.apply_intrinsic(resolution, intrinsic)
        return transformables



class ExtrinsicImage(Transform):
    
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
            if isinstance(transformable, (Image, ImageSegMask, ImageDepth)):
                if transformable.data['cam_id'] in self.cam_ids:
                    del_extrinsic = self.del_extrinsics[transformable.data['cam_id']]
                    transformable.apply_extrinsic(del_extrinsic)
        return transformables



class RandomExtrinsicImage(Transform):
    def __init__(self, *, prob=0.5, angles=[1, 1, 1], scope="frame", **kwargs):
        super().__init__(scope=scope)
        self.prob = prob
        self.angles = angles

    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
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
            if isinstance(transformable, (Image, ImageSegMask, ImageDepth)):
                del_extrinsic = (del_R, np.array([0, 0, 0]))
                transformable.apply_extrinsic(del_extrinsic)
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

    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
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
                    transformable.apply_extrinsic(del_extrinsic)
        
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

    def __call__(self, *transformables, seeds=[None, None, None], **kwargs):
        for seed in seeds:
            if self.scope in str(seed):
                random.seed(seed)
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
    RandomImageISP, 
    IntrinsicImage, 
    ExtrinsicImage, 
    RandomExtrinsicImage, 
    RandomRotationSpace, 
    RandomMirrorSpace, 
    RandomIntrinsicParam, 
    RandomExtrinsicParam, 
    ToTensor
]

for transform in available_transforms:
    TRANSFORMS.register_module(module=transform)
