import torch
import torch.nn as nn
from mmengine.model import BaseModule
from .utils import get_unproj_func
import numpy as np
from mmengine.registry import MODELS

@MODELS.register_module()
class Fish_BackprojectDepth(BaseModule):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width, intrinsic, fov=180):
        super(Fish_BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        cx, cy, fx, fy, p0, p1, p2, p3 = intrinsic
        unproj_func = get_unproj_func(p0, p1, p2, p3, fov=fov)
        uu, vv = np.meshgrid(
        np.linspace(0, self.width - 1, self.width), 
        np.linspace(0, self.height - 1, self.height)
        )
        x_distorted = (uu - cx) / fx
        y_distorted = (vv - cy) / fy
        r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
        theta = unproj_func(r_distorted)

        r_distorted[r_distorted < 1e-5] = 1e-5
        dd = np.sin(theta)
        xx = x_distorted * dd / r_distorted
        yy = y_distorted * dd / r_distorted
        zz = np.cos(theta)
        # ones = np.ones_like(zz)

        self.pix_coords = torch.unsqueeze(torch.from_numpy(np.stack([xx,yy,zz], axis=0)), 0) 
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height, self.width),
                                 requires_grad=False)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1, 1)
        self.pix_coords = torch.concat([self.pix_coords, self.ones], dim=1)
        self.pix_coords = nn.Parameter(self.pix_coords, requires_grad=False).to(torch.float32)

    def forward(self, depth):
        cam_points = depth * self.pix_coords.to(depth.device)

        return cam_points


@MODELS.register_module()
class Fish_Project3D(BaseModule):
    def __init__(self, batch_size, height, width, intrinsic, fov=180, init_cfg: dict | torch.List[dict] | None = None):
        super().__init__(init_cfg)
        
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.cx, self.cy, self.fx, self.fy, self.p0, self.p1, self.p2, self.p3 = intrinsic
        self.fov = fov

    def forward(self, camera_points, T):
        camera_points = torch.matmul(T[:, :3, :], camera_points.reshape(-1, 4, self.height * self.width))
        
        xx = camera_points[:, 0, ...]
        yy = camera_points[:, 1, ...]
        zz = camera_points[:, 2, ...]
        dd = torch.sqrt(xx**2 + yy**2)
        theta = torch.arctan2(dd, zz)
        fov_mask = torch.logical_and(theta >= -self.fov / 2 * torch.pi / 180, theta <= self.fov / 2 * torch.pi / 180)
        r_distorted = self.proj_func(theta, (self.p0, self.p1, self.p2, self.p3))
        uu = self.fx * (r_distorted * xx / dd) + self.cx
        vv = self.fy * (r_distorted * yy / dd) + self.cy
        uu[~fov_mask] = -1
        vv[~fov_mask] = -1
        
        return torch.stack([uu, vv], dim=-1).reshape(self.batch_size, self.height, self.width, 2)


    def proj_func(self, x, params):
        p0, p1, p2, p3 = params
        return x + p0 * x**3 + p1 * x**5 + p2 * x**7 + p3 * x**9