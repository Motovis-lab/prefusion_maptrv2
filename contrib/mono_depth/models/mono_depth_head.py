import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from prefusion.registry import MODELS
from mmengine.model import BaseModel
from contrib.fastbev_det.models.utils import transformation_from_parameters
import numpy as np


__all__ = ["Mono_Depth_Head", "SSIM"]

@MODELS.register_module()
class Mono_Depth_Head(BaseModel):
    def __init__(self,
                 fish_img_size=None,
                 pv_img_size=None, 
                 front_img_size=None,
                 min_depth=0.1,
                 max_depth=100,
                 downsample_factor: torch.Tensor = [16] , 
                 batch_size=None,
                 avg_reprojection=False,
                 disparity_smoothness=0.001,
                 depth_pose_net_cfg=dict(type="PoseDecoder", num_ch_enc=[128], num_input_features=2),
                 fish_unproject_cfg=None,
                 fish_project3d_cfg=None,
                 depth_decoder_conf=None,
                 ssim_cfg=dict(type='SSIM'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.depth_pose_net = MODELS.build(depth_pose_net_cfg)
        self.camera_type = ['fish', 'pv', 'front']
        self.frame_ids = None
        self.fish_img_size = fish_img_size
        self.pv_img_size = pv_img_size
        self.front_img_size = front_img_size
        self.min_depth = min_depth
        self.max_depth = max_depth  
        self.batch_size = batch_size
        self.avg_reprojection = avg_reprojection
        self.disparity_smoothness = disparity_smoothness
        self.each_camera_nums = None
        self.depth_decoder = MODELS.build(depth_decoder_conf)
        if isinstance(downsample_factor, int):
            self.downsample_factor = [downsample_factor]
        elif isinstance(downsample_factor, list):
            if len(downsample_factor)==1:
                self.downsample_factor = downsample_factor[0]
            else:
                self.downsample_factor = downsample_factor
        self.scales = np.sqrt(self.downsample_factor).astype(int)
        for camera_type in self.camera_type:
            if camera_type == 'fish':
                self.fish_backproject_depth = MODELS.build(fish_unproject_cfg)
                self.fish_project_3d = MODELS.build(fish_project3d_cfg)
            elif camera_type == 'pv':
                self.pv_backproject_depth = BackprojectDepth(self.batch_size * 5, self.pv_img_size[1], self.pv_img_size[0])
                self.pv_project_3d = Project3D(self.batch_size * 5, self.pv_img_size[1], self.pv_img_size[0])   
            else:
                self.front_backproject_depth = BackprojectDepth(self.batch_size * 1, self.front_img_size[1], self.front_img_size[0])
                self.front_project_3d = Project3D(self.batch_size * 1, self.front_img_size[1], self.front_img_size[0])
        self.ssim = MODELS.build(ssim_cfg)

    def forward(self, fish_features, pv_features, front_features, fish_inputs, pv_inputs, front_inputs):
        B_fish = fish_inputs[('color', 0)].shape[0]
        B_pv = pv_inputs[('color', 0)].shape[0]
        B_front = front_inputs[('color', 0)].shape[0]
        fish_all_features = fish_features.split(B_fish) # 按BATCH切分成 0 1 2  (B_fish个feature一组)
        pv_all_features = pv_features.split(B_pv)
        front_all_features = front_features.split(B_front)

        losses = dict()

        losses.update(self.forward_camera(fish_inputs, fish_all_features, "fish"))
        losses.update(self.forward_camera(pv_inputs, pv_all_features, "pv"))
        losses.update(self.forward_camera(front_inputs, front_all_features, "front"))
    
        return losses


    def forward_camera(self, inputs, all_features, camera_type):
        assert camera_type in self.camera_type, f"{camera_type} must be in {self.camera_type}."
        features = {}
        for i, k in enumerate(self.frame_ids):
            features[k] = all_features[i]
        
        outputs = dict()
        outputs['disp'] = self.depth_decoder([features[0]])[0]
        outputs.update(self.predict_poses(features))
        self.generate_images_pred(inputs, outputs, camera_type)
        losses = self.compute_losses(inputs, outputs, camera_type)

        return losses

    def predict_poses(self, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        for f_i in self.frame_ids[1:]:  # [1, 0, 2]
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 1:
                    pose_inputs = [features[f_i], features[1]]
                else:
                    pose_inputs = [features[1], features[f_i]]
            axisangle, translation = self.depth_pose_net(pose_inputs)
            outputs[("axisangle", f_i)] = axisangle
            outputs[("translation", f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 1))
        return outputs
    
    def generate_images_pred(self, inputs, outputs, camera_type):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        disp = outputs['disp']

        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

        outputs["depth"] = depth

        for i, frame_id in enumerate(self.frame_ids[1:]):
            T = outputs[("cam_T_cam", frame_id)]
            if camera_type == "fish":
                cam_points = getattr(self, f"{camera_type}_backproject_depth")(depth)
                pix_coords = getattr(self, f"{camera_type}_project_3d")(cam_points, T)
                outputs[("sample", frame_id)] = pix_coords
            else:
                cam_points = getattr(self,f"{camera_type}_backproject_depth")(depth, inputs["inv_K", 1])  # depth to camera point
                pix_coords = getattr(self, f"{camera_type}_project_3d")(cam_points, inputs["K", 1], T)
            
                outputs[("sample", frame_id)] = pix_coords

            outputs[("color", frame_id)] = F.grid_sample(
                inputs['color', frame_id],
                outputs[("sample", frame_id)],
                padding_mode="border")

            outputs[("color_identity", frame_id)] = \
                inputs['color', frame_id]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, camera_type):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        
        loss = 0
        disp = outputs['disp']
        reprojection_losses = []
        color = inputs['color', 1]
        target = inputs['color', 1]

        for frame_id in self.frame_ids[1:]:
            pred = outputs[("color", frame_id)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)
        
        # automasking
        identity_reprojection_losses = []
        for frame_id in self.frame_ids[1:]:
            pred = inputs["color", frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))  # ssim

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        # save both images, and do min all at once below
        identity_reprojection_loss = identity_reprojection_losses

        reprojection_loss = reprojection_losses

        # add random numbers to break ties  auto masking
        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape).cuda() * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        outputs["identity_selection"] = (
            idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.disparity_smoothness * smooth_loss

        losses["mono_loss_{}".format(camera_type)] = loss
        
        return losses


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

@MODELS.register_module()
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)