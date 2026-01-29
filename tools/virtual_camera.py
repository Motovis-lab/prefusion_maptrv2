import cv2
import yaml
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

from pathlib import Path
from functools import partial
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from copy import deepcopy
import cv2

#   [markdown]
# ## **Coordinate Systems:**
# | STYLE     | X-Y-Z              |
# | :-------: | :-----------------:|
# | MOTOVIS   | right-forward-up   |
# | openGL    | right-up-backward  |
# | camera     | right-down-forward |
# | pytorch3d | left-up-forward    |

 
def imshow(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

 
def get_extrinsic_from_euler(x, y, z, pitch, yaw, roll):
    R = Rotation.from_euler(
        'xyz', (pitch, yaw, roll), degrees=True
    ).as_matrix()
    t = np.float32([x, y, z])
    return R, t


def get_extrinsic_from_quant(x, y, z, qx, qy, qz, qw):
    R = Rotation.from_quant((qx, qy, qz, qw)).as_matrix()
    t = np.float32([x, y, z])
    return R, t


def get_intrinsic_mat(cx, cy, fx, fy):
    K = np.float32([
        [fx, 0, cx], 
        [0, fy, cy], 
        [0,  0,  1]
    ])
    return K


def proj_func(x, params):
    p0, p1, p2, p3 = params
    return x + p0 * x**3 + p1 * x**5 + p2 * x**7 + p3 * x**9


def poly_odd6(x, k0, k1, k2, k3, k4, k5):
    return x + k0 * x**3 + k1 * x**5 + k2 * x**7 + k3 * x**9 + k4 * x**11 + k5 * x**13


def get_unproj_func(p0, p1, p2, p3, fov=200):
    theta = np.linspace(-0.5 * fov * np.pi / 180,  0.5 * fov * np.pi / 180, 2000)
    theta_d = proj_func(theta, (p0, p1, p2, p3))
    params, pcov = curve_fit(poly_odd6, theta_d, theta)
    error = np.sqrt(np.diag(pcov)).mean()
    assert error < 1e-2, "poly parameter curve fitting failed: {:f}.".format(error)
    k0, k1, k2, k3, k4, k5 = params
    return partial(poly_odd6, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)


def ext_motovis2image(ext_motovis):
    x, y, z, pitch, yaw, roll = ext_motovis
    return x, -z, y, 90 + pitch, - roll, yaw


class BaseCamera:
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy, <distortion params>)
        - ego_mask : in shape (h, w)
        """
        self.resolution = resolution
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self._init_ext_int_mat()
        self.ego_mask = ego_mask
        self.camera_mask = None
    
    def _init_ext_int_mat(self):
        self.R_e, self.t_e = self.extrinsic
        self.T_e = np.eye(4)
        self.T_e[:3, :3] = self.R_e
        self.T_e[:3, 3] = self.t_e
        
        cx, cy, fx, fy = self.intrinsic[:4]
        self.K = np.float32([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0,  0,  1]
        ])
    
    def project_points_from_camera_to_image(self, camera_points):
        raise NotImplementedError

    def unproject_points_from_image_to_camera(self):
        raise NotImplementedError
    
    def get_camera_mask(self):
        """
        Returns a mask of the camera's view.
        """
        if self.camera_mask is None:
            self.camera_mask = self.ego_mask
        return self.camera_mask



class FisheyeCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, fov=None, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy, p0, p1, p2, p3)
        - fov : float, in degree
        - ego_mask : in shape (h, w)
        """
        super().__init__(resolution, extrinsic, intrinsic, ego_mask=ego_mask)
        if fov is None:
            self.fov = 225
        else:
            self.fov = fov

    def project_points_from_camera_to_image(self, camera_points, return_d=False):
        # camera_points in image-style: x-y-z right-down-forward
        cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
        xx = camera_points[0]
        yy = camera_points[1]
        zz = camera_points[2]
        # distance to camera center ray
        dd = np.sqrt(xx**2 + yy**2)
        # radius(focal=1) to light center point, aka theta between ray and center ray
        rr = theta = np.arctan2(dd, zz)
        # rr = theta = np.clip(np.arctan2(dd, zz), -self.fov / 2 * np.pi / 180, self.fov / 2 * np.pi / 180)
        fov_mask = np.logical_and(theta >= -self.fov / 2 * np.pi / 180, theta <= self.fov / 2 * np.pi / 180)
    
        # projected coords on fisheye camera image
        r_distorted = theta_distorted = proj_func(theta, (p0, p1, p2, p3))
        uu = np.float32(fx * (r_distorted * xx / dd) + cx)
        vv = np.float32(fy * (r_distorted * yy / dd) + cy)
        uu[~fov_mask] = -1
        vv[~fov_mask] = -1
        rho = np.linalg.norm(camera_points, axis=0)
        rho[~fov_mask] = 0
        if return_d:
            assert rho.min() >= 0, "rho should be non-negative"
            return uu, vv, rho
        else:
            return uu, vv


    def unproject_points_from_image_to_camera(self):
        W, H = self.resolution
        cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
        unproj_func = get_unproj_func(p0, p1, p2, p3, fov=self.fov)
        
        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W), 
            np.linspace(0, H - 1, H)
        )
        x_distorted = (uu - cx) / fx
        y_distorted = (vv - cy) / fy
        
        # r_distorted = theta_distorted
        r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
        # r_distorted[r_distorted < 1e-5] = 1e-5
        theta = unproj_func(r_distorted)
        # theta = np.clip(theta, - 0.5 * self.fov * np.pi / 180, 0.5 * self.fov * np.pi / 180)
        self.camera_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
    
        # get camera coords by ray intersecting with a sphere in image-style (x-y-z right-down-forward)
        r_distorted[r_distorted < 1e-5] = 1e-5
        dd = np.sin(theta)
        xx = x_distorted * dd / r_distorted
        yy = y_distorted * dd / r_distorted
        zz = np.cos(theta)
        
        camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)

        return camera_points
    

    def get_camera_mask(self, use_fov_mask=False):
        """
        Returns a mask of the camera's view.
        """
        if self.camera_mask is None and use_fov_mask:
            W, H = self.resolution
            cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
            unproj_func = get_unproj_func(p0, p1, p2, p3, fov=self.fov)
            
            uu, vv = np.meshgrid(
                np.linspace(0, W - 1, W), 
                np.linspace(0, H - 1, H)
            )
            x_distorted = (uu - cx) / fx
            y_distorted = (vv - cy) / fy
            
            # r_distorted = theta_distorted
            r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
            r_distorted[r_distorted < 1e-5] = 1e-5
            theta = unproj_func(r_distorted)
            self.camera_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
        
            if self.ego_mask is not None:
                self.camera_mask *= self.ego_mask
        else:
            self.camera_mask = self.ego_mask
    
        return self.camera_mask

    
    def _to_motovis_cfg(self):
        cfg_camera = {}
        cfg_camera['sensor_model'] = 'src.sensors.cameras.OpenCVFisheyeCamera'
        cfg_camera['image_size'] = self.resolution
        quant = Rotation.from_matrix(self.R_e).as_quat()
        cfg_camera['extrinsic'] = list(self.t_e) + list(quant)
        cfg_camera['pp'] = self.intrinsic[:2]
        cfg_camera['focal'] = self.intrinsic[2:4]
        cfg_camera['inv_poly'] = self.intrinsic[4:]
        cfg_camera['fov_fit'] = self.fov
        return cfg_camera


    @classmethod
    def init_from_motovis_cfg(cls, cfg_camera, use_default_fov=True):
        camera_model = cfg_camera['sensor_model']
        assert camera_model in ['src.sensors.cameras.OpenCVFisheyeCamera']

        resolution = cfg_camera['image_size']
        # ego system is in MOTOVIS-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy, p0, p1, p2, p3
        intrinsic = cfg_camera['pp'] + cfg_camera['focal'] + cfg_camera['inv_poly']
        if use_default_fov:
            fov = None
        else:
            fov = cfg_camera['fov_fit']

        return cls(resolution, extrinsic, intrinsic, fov)
class OCamModelCamera(BaseCamera):
    def __init__(self, extrinsic, intrinsic, fov=None, ego_mask=None):
        super().__init__(extrinsic, intrinsic, ego_mask=ego_mask)


    # def __init__(self, intrinsic, poly, invpoly, center_x, center_y, c, d, e, width, height):
    #     self.intrinsic = intrinsic  # 内参 (fx, fy, cx, cy)
    #     self.poly = poly  # 畸变正向多项式系数
    #     self.invpoly = invpoly  # 畸变反向多项式系数
    #     self.center_x = center_x  # 图像中心X坐标
    #     self.center_y = center_y  # 图像中心Y坐标
    #     self.c = c  # 畸变系数c
    #     self.d = d  # 畸变系数d
    #     self.e = e  # 畸变系数e
    #     self.width = width  # 图像宽度
    #     self.height = height  # 图像高度

    def project_points_from_camera_to_image(self, camera_points, return_d=False):
        # camera_points in image-style: x-y-z right-down-forward
        fx, fy, cx, cy = self.intrinsic  # 内参
        xx = camera_points[0]
        yy = camera_points[1]
        zz = camera_points[2]

        # 计算到相机中心的距离
        dd = np.sqrt(xx**2 + yy**2)
        
        # 计算 theta (视角)
        rr = np.arctan2(dd, zz)
        
        # 根据正向畸变多项式计算畸变系数
        r_distorted = self.proj_func(rr, self.poly)
        
        # 投影到图像坐标系
        uu = np.float32(fx * (r_distorted * xx / dd) + cx)
        vv = np.float32(fy * (r_distorted * yy / dd) + cy)
        
        # 判断是否在视场范围内
        fov_mask = np.logical_and(rr >= -np.pi / 4, rr <= np.pi / 4)  # 假设FOV为±45度
        uu[~fov_mask] = -1
        vv[~fov_mask] = -1
        
        # 计算距离
        rho = np.linalg.norm(camera_points, axis=0)
        rho[~fov_mask] = 0
        
        if return_d:
            assert rho.min() >= 0, "rho should be non-negative"
            return uu, vv, rho
        else:
            return uu, vv

    def proj_func(self, theta, poly):
        """
        使用OCam模型的正向畸变多项式进行投影
        """
        r_distorted = 0
        for i in range(len(poly)):
            r_distorted += poly[i] * (theta ** i)
        return r_distorted

    def inverse_project(self, u, v):
        """
        使用反向畸变多项式进行反投影
        """
        # 反向投影公式 (此处简化示例，实际根据你的`invpoly`系数实现)
        r_inv = self.inv_proj_func(u, v, self.invpoly)
        # 计算反向投影后的世界坐标
        # 这个过程将基于r_inv和其它内参来计算
        return r_inv

    def inv_proj_func(self, u, v, invpoly):
        """
        使用OCam模型的反向畸变多项式进行反投影
        """
        r_inv = 0
        for i in range(len(invpoly)):
            r_inv += invpoly[i] * (u ** i)  # 根据实际的反向多项式调整此公式
        return r_inv

class PerspectiveCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy)
        - ego_mask : in shape (h, w)
        """
        super().__init__(resolution, extrinsic, intrinsic, ego_mask=ego_mask)
    
    def unproject_points_from_image_to_camera(self):
        W, H = self.resolution
        cx, cy, fx, fy = self.intrinsic
        
        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W), 
            np.linspace(0, H - 1, H)
        )
        # get camera coords by ray intersecting with a z-plane in image-style (x-y-z right-down-forward)
        xx = (uu - cx) / fx
        yy = (vv - cy) / fy
        zz = np.ones_like(uu)

        camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)

        return camera_points

    def project_points_from_camera_to_image(self, camera_points, return_z=False, return_d=False):
        img_points = np.matmul(self.K, camera_points.reshape(3, -1)).reshape(camera_points.shape)
        img_points[2, np.abs(img_points[2]) < 1e-5] = 1e-5
        uu = np.float32(img_points[0] / img_points[2])
        vv = np.float32(img_points[1] / img_points[2])
        if return_z:
            return uu, vv, img_points[2]
        elif return_d:
            return uu, vv, np.linalg.norm(camera_points, axis=0)
        else:
            return uu, vv
    
    def _to_motovis_cfg(self):
        cfg_camera = {}
        cfg_camera['sensor_model'] = 'src.sensors.cameras.PerspectiveCamera'
        cfg_camera['image_size'] = self.resolution
        quant = Rotation.from_matrix(self.R_e).as_quat()
        cfg_camera['extrinsic'] = list(self.t_e) + list(quant)
        cfg_camera['pp'] = self.intrinsic[:2]
        cfg_camera['focal'] = self.intrinsic[2:4]
        return cfg_camera

    @classmethod
    def init_from_nuscense_cfg(cls, cfg):
        width = cfg['width']
        height = cfg['height']
        resolution = [width, height]
        t_e = cfg['calibrated_sensor']['translation']
        quat = cfg['calibrated_sensor']['rotation']
        R_nus = Rotation.from_euler("xyz", angles=(0,0,-90), degrees=True).as_matrix()
        R_e = R_nus.T @ Rotation.from_quat((quat[1], quat[2], quat[3], quat[0])).as_matrix()
        # R_e = Quaternion(quat).rotation_matrix
        extrinsic = (R_e, t_e)
        intrinsic_m = cfg['calibrated_sensor']['camera_intrinsic']
        intrinsic = [intrinsic_m[0][2], intrinsic_m[1][2], intrinsic_m[0][0], intrinsic_m[1][1]]
        
        return cls(resolution, extrinsic, intrinsic)
    
    @classmethod
    def init_from_av2_cfg(cls, cfg):
        pass

    @classmethod
    def init_from_motovis_cfg(cls, cfg_camera):
        camera_model = cfg_camera['sensor_model']
        assert camera_model in [
            'src.sensors.cameras.PerspectiveCamera', 
            'src.sensors.cameras.PinholeCamera',
            'src.sensors.cameras.DDADPerspectiveCamera',
            'src.sensors.cameras.NuScenesPerspectiveCamera'
        ]

        resolution = cfg_camera['image_size']
        # ego system is in MOTOVIS-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy
        intrinsic = cfg_camera['pp'] + cfg_camera['focal']

        return cls(resolution, extrinsic, intrinsic)

 
AVAILABLE_CAMERA_TYPES = [FisheyeCamera, PerspectiveCamera, OCamModelCamera]



def _check_camera_type(camera):
    return any([isinstance(camera, camera_type) for camera_type in AVAILABLE_CAMERA_TYPES])


def render_image(src_img, src_camera, dst_camera):
    assert _check_camera_type(src_camera), 'AssertError: src_camera must be one of {}'.format(AVAILABLE_CAMERA_TYPES)
    assert _check_camera_type(dst_camera), 'AssertError: dst_camera must be one of {}'.format(AVAILABLE_CAMERA_TYPES)

    # assert src_img.shape[:2][::-1] == src_camera.resolution, 'AssertError: src_image must have the same resolution as src_camera'

    R_e_src = src_camera.R_e
    R_e_dst = dst_camera.R_e

    R_dst_src = R_e_dst.T @ R_e_src

    dst_camera_points = dst_camera.unproject_points_from_image_to_camera()

    rot_dst_camera_points = R_dst_src.T @ dst_camera_points

    uu, vv = src_camera.project_points_from_camera_to_image(rot_dst_camera_points)
    src_camera_mask = src_camera.get_camera_mask()

    dst_img = cv2.remap(
        src_img, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=cv2.INTER_LINEAR
    )
    
    src_img_mask = np.ones(src_img.shape[:2], dtype=np.float32)
    if src_camera_mask is not None:
        src_img_mask = cv2.resize(src_img_mask, (768, 512), interpolation=cv2.INTER_NEAREST)
        src_img_mask *= src_camera_mask
    dst_img_mask = cv2.remap(
        src_img_mask, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=cv2.INTER_LINEAR
    )

    return dst_img, dst_img_mask

 
def create_virtual_perspective_camera(resolution, euler_angles, transitions, intrinsic='auto'):
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
    # ego system, in motovis-style, x-y-z right-forward-up
    R = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    t = transitions
    return PerspectiveCamera(resolution, (R, t), intrinsic)

def create_nus_virtual_perspective_camera(resolution, euler_angles, transitions, intrinsic='auto'):
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
    # ego system, in motovis-style, x-y-z right-forward-up
    R_nus = Rotation.from_euler("xyz", angles=(0,0,-90), degrees=True).as_matrix()
    R = R_nus @ Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix() @ Rotation.from_euler("xyz", angles=(0,0,-57), degrees=True).as_matrix()
    t = transitions
    return PerspectiveCamera(resolution, (R, t), intrinsic)


def create_virtual_fisheye_camera(resolution, euler_angles, transitions, intrinsic='auto', fov=225):
    # inv_poly: [0.05345955558134785, -0.005850248788053312, -0.0005388425917994607, -0.0001609567223788042]
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
    # ego system, in motovis-style, x-y-z right-forward-up
    R = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    t = transitions
    return FisheyeCamera(resolution, (R, t), intrinsic, fov=fov)


 
def project_points_fisheye(points: np.ndarray, cfg) -> np.ndarray:
    EPS_FLOAT32 = float(np.finfo(np.float32).eps)
    xc = points[:, 0]
    yc = points[:, 1]
    zc = points[:, 2]
    norm = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(norm, zc)
    fov_mask = theta > cfg.fov_fit / 2 * np.pi / 180
    rho = (
        theta
        + cfg.inv_poly[0] * theta**3
        + cfg.inv_poly[1] * theta**5
        + cfg.inv_poly[2] * theta**7
        + cfg.inv_poly[3] * theta**9
    )
    width, height = cfg.image_size
    image_radius = np.sqrt((width / 2) ** 2 + (height) ** 2)
    rho[fov_mask] = 2 * image_radius / cfg.focal[0]
    xn = rho * xc / norm
    yn = rho * yc / norm
    xn[norm < EPS_FLOAT32] = 0
    yn[norm < EPS_FLOAT32] = 0
    norm_coords = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    intrinsic_mat = np.array(
        [
            [cfg.focal[0], 0, cfg.pp[0]],
            [0, cfg.focal[1], cfg.pp[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    image_coords = norm_coords @ intrinsic_mat.T
    return image_coords[:, :2]

def project_points_perspective(points: np.ndarray, cfg) -> np.ndarray:
    EPS_FLOAT32 = float(np.finfo(np.float32).eps)
    xc = points[:, 0]
    yc = points[:, 1]
    zc = points[:, 2]
    norm = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(norm, zc)
    fov_mask = theta > cfg.fov_fit / 2 * np.pi / 180
    rho = (
        theta
        + cfg.inv_poly[0] * theta**3
        + cfg.inv_poly[1] * theta**5
        + cfg.inv_poly[2] * theta**7
        + cfg.inv_poly[3] * theta**9
    )
    width, height = cfg.image_size
    image_radius = np.sqrt((width / 2) ** 2 + (height) ** 2)
    rho[fov_mask] = 2 * image_radius / cfg.focal[0]
    xn = rho * xc / norm
    yn = rho * yc / norm
    xn[norm < EPS_FLOAT32] = 0
    yn[norm < EPS_FLOAT32] = 0
    norm_coords = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    intrinsic_mat = np.array(
        [
            [cfg.focal[0], 0, cfg.pp[0]],
            [0, cfg.focal[1], cfg.pp[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    image_coords = norm_coords @ intrinsic_mat.T
    return image_coords[:, :2]

def plot_points_on_image(image, points, color, radius=2):
    im_size = image.shape[:2][::-1]
    for i in range(len(points)):
        if points[i, 0] < 0 or points[i, 0] >= im_size[0]:
            continue
        if points[i, 1] < 0 or points[i, 1] >= im_size[1]:
            continue
        cv2.circle(image, (int(round(points[i, 0])), int(round(points[i, 1]))), radius, color, -1,)

 
def load_point_cloud(point_cloud_path: Path, discard_intensity=True, filter_zero_intensity=True) -> np.ndarray:
    if point_cloud_path.suffix == ".bin":
        point_cloud = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported point cloud file format: {point_cloud_path.suffix}")
    if filter_zero_intensity:
        valid_intensity = point_cloud[:, 3] > 0  # Check if intensity is positive
        point_cloud = point_cloud[valid_intensity]
    if discard_intensity:
        point_cloud = point_cloud[:, :3]
    return point_cloud

 
import open3d as o3d
def pcd_lidar_point(save_path, lidar_points):
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    points_intensities = lidar_points[:, 3][:, None]
    points_positions = lidar_points[:,:3]
    lidar_map = o3d.t.geometry.PointCloud(device)
    ego_front = 3.8168
    ego_left = 1.0677
    ego_back = -1.0584
    ego_right = -1.0802
    ego_top = 1.9012
    mask_front_back = np.logical_or((points_positions[..., 0] > ego_front), (points_positions[..., 0] < ego_back))
    mask_right_left = np.logical_or((points_positions[..., 1] < ego_right), (points_positions[..., 1] > ego_left))
    mask_top = points_positions[..., 2] > ego_top
    mask = np.logical_or(np.logical_or(mask_front_back, mask_right_left), mask_top)
    points_positions = points_positions[mask, ...] 
    points_intensities = points_intensities[mask, ...]
    lidar_map.point.positions = o3d.core.Tensor(points_positions , dtype, device)
    lidar_map.point.intensity = o3d.core.Tensor(points_intensities , dtype, device)
    lidar_map = lidar_map.voxel_down_sample(voxel_size=0.1)
    o3d.t.io.write_point_cloud(str(save_path), lidar_map)

def read_pcd_lidar(lidar_path):
    lidar_data = o3d.t.io.read_point_cloud(lidar_path)
    lidar_points_positions = lidar_data.point.positions.numpy()
    lidar_points_intensity = lidar_data.point.intensity.numpy()
    lidar_points = np.concatenate([lidar_points_positions, lidar_points_intensity], axis=1)
    return lidar_points

 
def plot_points_on_image_r(image, points, colors=128, radius=1):
    im_size = image.shape[:2][::-1]
    for i in range(len(points)):
        if points[i, 0] < 0 or points[i, 0] >= im_size[0]:
            continue
        if points[i, 1] < 0 or points[i, 1] >= im_size[1]:
            continue
        if len(colors) != len(points):
            color = colors
        else:
            color = (colors[i] / colors.max() * 255).astype(np.uint8)
        cv2.circle(image, (int(round(points[i, 0])), int(round(points[i, 1]))), radius, [int(color), int(color), int(color), 0], -1,)
    return image

def plot_points_on_image_depth(image, depth, points, colors=128, radius=1):
    im_size = image.shape[:2][::-1]
    for i in range(len(points)):
        if points[i, 0] < 0 or points[i, 0] >= im_size[0]:
            continue
        if points[i, 1] < 0 or points[i, 1] >= im_size[1]:
            continue
        if colors[i] < 0:
            continue
        if len(colors) != len(points):
            color = colors
        else:
            color = (colors[i] / colors.max() * 255).astype(np.uint8)
        cv2.circle(image, (int(round(points[i, 0])), int(round(points[i, 1]))), radius, [int(color), 255- int(color), int(color), 255], -1,)
        assert colors[i] >= 0, "colors values shuble be >=0"
        depth[int(round(points[i, 1]))-1, int(round(points[i, 0]))-1] = colors[i]
    return image, depth

def render_image_with_src_camera_points(src_image, src_camera, dst_camera, lidar_point, colormap=None, return_depth=False):
    # assert src_image.shape[:2][::-1] == src_camera.resolution, 'AssertError: src_image must have the same resolution as src_camera'

    R_e_src = src_camera.R_e
    R_e_dst = dst_camera.R_e

    R_dst_src = R_e_dst.T @ R_e_src

    dst_camera_points = dst_camera.unproject_points_from_image_to_camera()
    # dst_camera_points *= dst_camera_mask.reshape(1, -1)
    # print(dst_camera_points.shape)

    
    rot_dst_camera_points = R_dst_src.T @ dst_camera_points

    uu, vv = src_camera.project_points_from_camera_to_image(rot_dst_camera_points)
    src_camera_mask = src_camera.get_camera_mask()

    dst_img = cv2.remap(
        src_image, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=cv2.INTER_LINEAR
    )
    src_img_mask = np.ones(src_image.shape[:2], dtype=np.float32)
    if src_camera_mask is not None:
        src_img_mask *= src_camera_mask
    dst_img_mask = cv2.remap(
        src_img_mask, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=cv2.INTER_LINEAR
    )
    uu, vv, zz = dst_camera.project_points_from_camera_to_image(lidar_point[:3, ...], return_d=True)
    # print(uu.shape, vv.shape)
    image_coords = np.stack([uu, vv], axis=1)
    depth = (np.zeros_like(dst_img)[..., 0]).astype(np.float32) - 1
    dst_img_w_point, depth = plot_points_on_image_depth(deepcopy(dst_img), depth, image_coords, zz)
    if return_depth:
        return dst_img, dst_img_w_point, dst_img_mask, depth
    else:
        return dst_img, dst_img_w_point, dst_img_mask

 
def fish_unproject_points_from_image_to_camera(intrinsic, fov, ogfH, ogfW, H, W):
    cx, cy, fx, fy, p0, p1, p2, p3 = intrinsic
    unproj_func = get_unproj_func(p0, p1, p2, p3, fov=fov)
    
    uu, vv = np.meshgrid(
        np.linspace(0, ogfW - 1, W), 
        np.linspace(0, ogfH - 1, H)
    )
    x_distorted = (uu - cx) / fx
    y_distorted = (vv - cy) / fy
    
    # r_distorted = theta_distorted
    r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
    # r_distorted[r_distorted < 1e-5] = 1e-5
    theta = unproj_func(r_distorted)
    # theta = np.clip(theta, - 0.5 * self.fov * np.pi / 180, 0.5 * self.fov * np.pi / 180)

    # get camera coords by ray intersecting with a sphere in image-style (x-y-z right-down-forward)
    r_distorted[r_distorted < 1e-5] = 1e-5
    dd = np.sin(theta)
    xx = x_distorted * dd / r_distorted
    yy = y_distorted * dd / r_distorted
    zz = np.cos(theta)

    return xx, yy, zz

def pv_unproject_points_from_image_to_camera(intrinsic, ogfH, ogfW, H, W):
    cx, cy, fx, fy = intrinsic
    
    uu, vv = np.meshgrid(
        np.linspace(0, ogfW - 1, W), 
        np.linspace(0, ogfH - 1, H)
    )
    # get camera coords by ray intersecting with a z-plane in image-style (x-y-z right-down-forward)
    xx = (uu - cx) / fx
    yy = (vv - cy) / fy
    zz = np.ones_like(uu)

    return xx, yy, zz

# import cv2
# import yaml
# import numpy as np

# import matplotlib.pyplot as plt

# from pathlib import Path
# from functools import partial
# from scipy.optimize import curve_fit
# from scipy.spatial.transform import Rotation
# from pyquaternion import Quaternion
# from mmcv import Config
# from copy import deepcopy
# import cv2
# from virtual_camera import FisheyeCamera, read_pcd_lidar, render_image_with_src_camera_points, imshow, create_virtual_fisheye_camera


# yaml_file = '../test/show_data/calibration_back.yml'
# cfgs = yaml.safe_load(open(yaml_file, 'r'))
# vcamera_fisheye_left = create_virtual_fisheye_camera((1024, 640), (-135, 0, 0), (0,0,0))
# vcamera_fisheye_left = create_virtual_fisheye_camera((1024, 640), (-135, 0, -180), (0,0,0))
# R_nus = Rotation.from_euler("xyz", angles=(0,0,90), degrees=True).as_matrix()
# cfgs['rig']['camera11']['extrinsic'] = [*((R_nus.T @ np.array(cfgs['rig']['camera11']['extrinsic'][:3]).reshape(3)).tolist()), *(Rotation.from_matrix((R_nus.T @ Rotation.from_quat(cfgs['rig']['camera11']['extrinsic'][3:]).as_matrix())).as_quat().tolist())]
# # print(cfgs['rig']['camera5']['extrinsic'][:3])
# camera5 = FisheyeCamera.init_from_motovis_cfg(cfgs['rig']['camera11'])

# dummy_lidar_point_match = read_pcd_lidar("/ssd4/home/wuhan/MatrixVT_tda4/tools_data/mtv4d/test_ori.pcd").T
# intensity = deepcopy(dummy_lidar_point_match[3, :])
# dummy_lidar_point_match[3, :] = 1

# # (-135, 0, 90), (-1, 2, 1)
# cam8_R_T = np.eye(4)  
# R_Matrix = ((Quaternion(axis=[0,0,1], angle=(0)*np.pi/180) * Quaternion(axis=[1,0,0], angle=(-135*np.pi/180))*Quaternion(axis=[0,1,0], angle=0)).rotation_matrix).T
# # R_Matrix = Rotation.from_euler("XYZ", angles=(-135, 0, 0), degrees=True).as_matrix().T
# cam8_R_T[:3, :3] = R_Matrix
# cam8_R_T[:3, 3]  = -R_Matrix @ np.array(cfgs['rig']['camera5']['extrinsic'][:3])
# # lidar的坐标系是右前上，需转到前左上， vcamera是基于前左上的坐标系
# # tmp = np.eye(4)
# # tmp[:3, :3] = R_nus
# # dummy_lidar_point_match = tmp.T @ dummy_lidar_point_match
# vcamera_fisheye_left_lidar = cam8_R_T @ deepcopy(dummy_lidar_point_match)
# vcamera_fisheye_left_lidar[3, :] = intensity

# src_image = plt.imread('../test/show_data/camera_data/camera11/1698826056864.jpg')
# # imshow(src_image)

# dst_image, dst_img_w_point, dst_mask = render_image_with_src_camera_points(src_image, camera5, vcamera_fisheye_left, vcamera_fisheye_left_lidar)
# # imshow(dst_image)
# imshow(dst_img_w_point)
