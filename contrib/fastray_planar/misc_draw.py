import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtv4d import draw_points
from mtv4d.annos_4d.misc import boxes_to_corners_3d
from mtv4d.utils.box_base import to_corners_7, box_corners_to_dot_cloud
from mtv4d.utils.geo_base import transform_pts_with_T
from mtv4d.utils.draw_base import draw_boxes
from mtv4d.utils.sensors import FisheyeCameraModel  # same as copious
import copy
from prefusion import SegIouLoss, DualFocalLoss, PlanarBbox3D, PlanarRectangularCuboid, PlanarSquarePillar, \
    PlanarOrientedCylinder3D, PlanarCylinder3D, PlanarParkingSlot3D
import pdb
import torch
import torch.nn.functional as F
import os

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1 
    color = (color * 255).astype(np.uint8)
    return color

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]
    img = img*255
    img = img.astype(np.uint8)
    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))

    size = np.zeros(2).astype(int)
    size[0] = int(img.shape[0] // num_rows)    # height
    size[1] = int(img.shape[1]// num_rows)    # width

    # canvas: 高度按 num_rows * height，宽度 = 原图宽 + 每个heatmap列 * 宽度
    full_img = np.zeros((size[0] * num_rows, size[1] * (num_cols + 1), 3), np.uint8)
    inp_small = cv2.resize(img, (size[1], size[0]))
    full_img[:size[0], :size[1]] = inp_small

    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = cv2.resize(out[part_idx], (size[1], size[0])).astype(float)

        out_img = inp_small.copy() * 0.6
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * 0.4

        row_offset = (i // num_cols) * size[0]
        col_offset = (i % num_cols + 1) * size[1]  # +1 是为了不覆盖原图
        full_img[row_offset:row_offset + size[0], col_offset:col_offset + size[1]] = out_img


    return full_img

def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0, 0, 0]), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 8)):
        # print(n)
        # print(inputs[n].shape)
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )

    return np.concatenate(batch_img)

class FisheyeToIPM:
    def __init__(self, img_size=(640, 384), camera_model=None, extrinsic=None, ipm_width=1280,ipm_height=1280,scale=0.01875, device='cuda'):
        self.w, self.h = img_size
        self.camera_model = camera_model  #mvt4d相机模型
        self.extrinsic = extrinsic  # 3x4
        self.device = device

        # 输出IPM尺寸
        self.ipm_width = ipm_width
        self.ipm_height = ipm_height
        self.scale = scale  # 每像素表示的米数
        # 创建映射表
        self.map_grid = self.create_mapping()

    def distort_point(self, x_n, y_n):
        cx, cy = self.camera_model.pp
        fx, fy = self.camera_model.focal
        p0, p1, p2, p3 = self.camera_model.inv_poly
        r = torch.sqrt(x_n**2 + y_n**2)
        theta = torch.atan(r)
        theta_d = theta * (1 + p0*theta**2 + p1*theta**4 + p2*theta**6 + p3*theta**8)
        scale = torch.where(r > 0, theta_d / r, torch.ones_like(r))
        x_dist = x_n * scale
        y_dist = y_n * scale
        x_pixel = x_dist * fx + cx
        y_pixel = y_dist * fy + cy
        return x_pixel, y_pixel

    def create_mapping(self):
        offset_x = self.ipm_width // 2
        offset_y = self.ipm_height // 2

        # 创建IPM平面网格
        u = torch.linspace(0, self.ipm_width - 1, self.ipm_width, device=self.device)
        v = torch.linspace(0, self.ipm_height - 1, self.ipm_height, device=self.device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')

        Xw = (grid_u - offset_x) * self.scale
        Yw = (grid_v - offset_y) * self.scale
        Zw = torch.zeros_like(Xw)

        world_points = torch.stack((Xw, Yw, Zw), dim=-1)  # (H, W, 3)

        R = torch.tensor(self.extrinsic[:, :3], device=self.device, dtype=torch.float32)
        t = torch.tensor(self.extrinsic[:, 3], device=self.device, dtype=torch.float32)

        # 世界坐标 -> 相机坐标
        cam_points = torch.matmul(world_points - t, R)

        x_n = cam_points[..., 0] / (cam_points[..., 2] + 1e-6)
        y_n = cam_points[..., 1] / (cam_points[..., 2] + 1e-6)

        # 只处理在相机前方的点
        valid = cam_points[..., 2] > 0

        # 添加鱼眼畸变
        x_dist, y_dist = self.distort_point(x_n, y_n)

        # 归一化到 [-1, 1]，供grid_sample使用
        map_x = (2 * x_dist / (self.w - 1)) - 1
        map_y = (2 * y_dist / (self.h - 1)) - 1

        map_grid = torch.stack((map_x, map_y), dim=-1)

        # 把无效区域设成 (-2, -2)，保证采样不到
        map_grid[~valid] = -2

        return map_grid

    def transform_image(self, img):
        """
        :param img: 输入图像 (H, W, C) numpy array or torch tensor
        :return: OpenCV格式输出图像 (H, W, C) uint8
        """
        if isinstance(img, torch.Tensor):
            img_tensor = img.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, C, H, W)
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.

        # grid_sample
        output = F.grid_sample(
            img_tensor, 
            self.map_grid.unsqueeze(0), 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )  # (1, C, ipm_h, ipm_w)

        # 转成 numpy (H, W, C)
        output = output.squeeze(0).permute(1, 2, 0)  # (ipm_h, ipm_w, C)
        output = torch.clamp(output, 0, 1)  # 防止溢出
        output = (output * 255).byte().cpu().numpy()  # 回到 CPU，变成 uint8

        return output


def fisheye_to_ipm(image, camera_model, extrinsic, ipm_width=1280,ipm_height=1280,scale=0.01875, device='cuda'):
    """
    单路鱼眼图像到IPM鸟瞰图
    :param image: 输入图像 (H, W, C)
    :param camera_model: mvt4d相机模型
    :param extrinsic: 外参参数
    :param device: 'cuda' or 'cpu'
    :return: (C, ipm_height, ipm_width) tensor
    """
    transformer = FisheyeToIPM(img_size=(image.shape[1], image.shape[0]),
                                camera_model=camera_model,
                                extrinsic=extrinsic,
                                ipm_width=1280,
                                ipm_height=1280,
                                scale=0.01875,
                                device=device)
    ipm_image = transformer.transform_image(image)
    return ipm_image

def generate_camera_fusion_mask(ipm_size, ipm_range, angle=45, fusion_angle=15):
    # ipm_size, ipm_range hw=1600, angle=75, fusion_angle=15
    # warning hw is useless
    hw = 1600
    # angle = 75
    # fusion_angle = 15

    h, w = ipm_size, ipm_size

    rad75 = np.deg2rad(angle)
    small_angle = np.deg2rad(fusion_angle)  #
    # ----------------- set the range
    scope = int(1.80 / ipm_range * ipm_size)
    car_x = int(1.90 / ipm_range * ipm_size)
    car_y = int(5.00 / ipm_range * ipm_size)

    front = np.zeros([h, w])
    p1 = (w / 2 - car_x / 2, h / 2 - car_y / 2)  #
    p11 = (w / 2, p1[1])
    x = scope / 2 / np.sin(small_angle)
    h_ = np.cos(np.deg2rad(90) - rad75 - small_angle) * x
    w_ = np.sin(np.deg2rad(90) - rad75 - small_angle) * x
    p2 = (p1[0] - w_, p1[1] - h_)
    p3 = (p2[0] - p2[1] * np.tan(np.deg2rad(90) - rad75), 0)
    pts = np.array([p1, p2, p3, (w / 2, 0), p11]).astype('int')
    cv2.fillPoly(front, [pts], 1)
    # plt.imshow(front), plt.show()

    # ------------------------- 2 平行区域
    h__ = np.cos(np.deg2rad(90) - rad75 + small_angle) * x
    w__ = np.sin(np.deg2rad(90) - rad75 + small_angle) * x
    p22 = (p1[0] - w__, p1[1] - h__)
    if p22[0] - p22[1] * np.tan(np.deg2rad(90) - rad75) > 0:  # 上边缘
        p32 = (p22[0] - p22[1] * np.tan(np.deg2rad(90) - rad75), 0)
        pts = np.array([p2, p3, p32, p22]).astype('int')
    else:
        p32 = (0, p22[1] - p22[0] * np.tan(rad75))
        pts = np.array([p2, p3, (0, 0), p32, p22]).astype('int')
    mask = np.zeros_like(front)
    cv2.fillPoly(mask, [pts], 1)
    # plt.imshow(mask+front), plt.show()
    py, px = np.nonzero(mask)
    d1 = np.array([px, py]).T - np.array(p2)
    d2 = np.array([-np.cos(rad75), -np.sin(rad75)])
    dd1 = np.linalg.norm(d1, axis=1)
    q = dd1 * np.sin(np.arccos(d1 @ d2 / dd1))
    q = np.clip(q, a_min=0, a_max=scope)
    front[mask == 1] = 1 - q / scope
    # plt.imshow(front), plt.show()

    # ------------------------- 3 小三角
    pts = np.array([p1, p2, p22]).astype('int')
    mask = np.zeros_like(front)
    cv2.fillPoly(mask, [pts], 1)
    py, px = np.nonzero(mask)
    d1 = np.array([px, py]).T - np.array(p1)
    d1 = d1 / (np.linalg.norm(d1, axis=1).reshape(-1, 1) + 1e-8)
    d2 = np.array([-np.cos(rad75 - small_angle), -np.sin(rad75 - small_angle)])
    q = np.arccos(d1 @ d2)
    q = np.clip(q, a_min=0, a_max=small_angle * 2)
    front[mask == 1] = q / small_angle / 2
    front = np.clip(front + front[:, ::-1], 0, 1)
    # plt.imshow(front), plt.show()
    back = cv2.rotate(front, cv2.ROTATE_180)

    left = np.zeros_like(front)
    pts = np.array([p1, p2, p3, (0, 0), (0, h / 2), (p1[0], h / 2)]).astype('int')
    cv2.fillPoly(left, [pts], 1)
    left[(left > 0) * (front > 0) * (front < 1)] = 1 - front[(left > 0) * (front > 0) * (front < 1)]
    left = np.clip(left + left[::-1], 0, 1)
    right = cv2.rotate(left, cv2.ROTATE_180)
    # left = cv2.rotate(front, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # right = cv2.rotate(front, cv2.ROTATE_90_CLOCKWISE)
    # self.ipm_mask_dict = {
    #     'front': back,
    #     'left': left,
    #     'rear': front,
    #     'right': right
    # }
    ipm_mask_dict = {
        'camera_sv_front': back,
        'camera_sv_left': left,
        'camera_sv_rear': front,
        'camera_sv_right': right
    }
    return ipm_mask_dict


def draw_every_element():
    # get ---------------------------------
    for i, branch in enumerate(pred_dict.keys()):
        pred_dict_branch = pred_dict[branch]
        # gt_dict_branch = gt_dict[branch]
        pred_dict_branch_0 = {**pred_dict_branch}
        for key in pred_dict_branch_0:
            if key in ['cen', 'seg']:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
            else:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
        transformables = batched_input_dict['transformables'][0]
        # scene_frame_id = batched_input_dict['index_infos'][0].scene_frame_id
        tensor_smith = transformables[branch].tensor_smith
        voxel_range = tensor_smith.voxel_range

        match tensor_smith:
            case PlanarBbox3D():  # | PlanarRectangularCuboid() | PlanarSquarePillar() | PlanarOrientedCylinder3D():
                # draw this onto the image.
                gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor
                results = tensor_smith.reverse(gt)
                # pred = tensor_smith.reverse(pred_dict_branch_0)
                for box in results:
                    corners_ego = pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation'])
                    for k, v in result_dict.items():
                        img, camera_model, Tce = result_dict[k]
                        corners_2d = draw_camera_points_to_image_points(camera_model,
                                                                        transform_pts_with_T(corners_ego, Tce))
                        draw_boxes(img, corners_2d)
                cv2.imwrite(f'/tmp/1234/result_{cam}.jpg', img)
                exit()
    return


def get_fisheye_camera_model_from_ext_int(camera_dict, cam_name):
    cam_rigs = {
        'pp': camera_dict['intrinsic'][:2],  #  [319.5, 191.5, 160.0, 160.0, 0.1, 0, 0, 0]
        'focal': camera_dict['intrinsic'][2:4],
        'inv_poly': camera_dict['intrinsic'][4:8],
        'image_size': [512, 768],
        'fov_fit': 190
    }
    calib = {'rig': {cam_name: cam_rigs}}
    camera_model = FisheyeCameraModel(calib, cam_name)
    return camera_model


def get_fisheye_camera_model(camera_image, cam_name):
    cam_rigs = {
        'pp': camera_image.intrinsic[:2],  #  [319.5, 191.5, 160.0, 160.0, 0.1, 0, 0, 0]
        'focal': camera_image.intrinsic[2:4],
        'inv_poly': camera_image.intrinsic[4:8],
        'image_size': camera_image.img.shape[1:],
        'fov_fit': 190
    }
    calib = {'rig': {cam_name: cam_rigs}}
    camera_model = FisheyeCameraModel(calib, cam_name)
    return camera_model
    # return camera_model.project_points(points)


def Rt2T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pred_planar_box_to_3d_corners(size, rotation, translation):
    # front left up
    x, y, z = translation
    a, b, c = np.array(size)/2
    corners = np.array(
        [
            [+a, -b, -c],
            [+a, +b, -c],
            [+a, +b, +c],
            [+a, -b, +c],
            [-a, -b, -c],
            [-a, +b, -c],
            [-a, +b, +c],
            [-a, -b, +c],
        ]
    )

    R = rotation
    corners = corners @ R.T
    corners += np.array([x, y, z]).reshape(-1, 3)
    return corners


def draw_camera_points_to_image_points(camera_model, points):
    return camera_model.project_points(points)


def draw_gt_pkl(im_path, cam_calib, boxes):
    img = cv2.imread(im_path)
    # cam_name = 'VCAMERA_FISHEYE_FRONT'
    cam_name = 'camera8'
    camera_model = get_fisheye_camera_model_from_ext_int(cam_calib, cam_name)
    Tce =  np.linalg.inv(Rt2T(cam_calib['extrinsic'][0], cam_calib['extrinsic'][1]))
    for box in boxes:
        corners_ego = pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation'])  # the bug maybe
        corners_2d = draw_camera_points_to_image_points(camera_model, transform_pts_with_T(corners_ego, Tce))
        draw_boxes(img, corners_2d)
    cv2.imwrite(f'/tmp/1234/result_{cam_name}.jpg', img)


def draw_points_with_label(points, boxes, save_path=None):
    # box_list = np.concatenate([to_corners_9(anno_box_to_9_values_box(bx)) for bx in box_objs]).reshape(-1, 3)
    corners_lidar = boxes.reshape(-1, 3)
    box_points = box_corners_to_dot_cloud(corners_lidar)
    pp = np.concatenate([points[:, :3], box_points])
    save_path = f'/tmp/1234/kuang0.ply' if save_path is None else save_path
    draw_points(pp, save_path)

def draw_results_parking_IPM(pred_dict, batched_input_dict, save_im=True):
    # get the result image here

    ipm_width = 1280
    ipm_height = 1280
    scale = 0.01875  # 每像素表示的米数
    result_dict = {}
    gt = batched_input_dict['transformables'][0]['parkingslot_3d'].tensor

    for cam, v in batched_input_dict['transformables'][0]['camera_images'].transformables.items():
        img0 = v.tensor['img']
        img = img0.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 128
        camera_model = get_fisheye_camera_model(v, cam)
        Tce = np.linalg.inv(Rt2T(v.extrinsic[0], v.extrinsic[1]))
        result_dict[cam] = (img, camera_model, Tce, v)

    branch = 'parkingslot_3d'
    # gt
    transformables = batched_input_dict['transformables'][0]
    tensor_smith = transformables[branch].tensor_smith
    voxel_range = tensor_smith.voxel_range
    results = tensor_smith.reverse(gt)
    # draw results using tensorsmith reverse 
    
    ipm_imgs=[]
    for k, v in result_dict.items():
        img, camera_model, Tce, v = result_dict[k]

        extrinsic=Rt2T(v.extrinsic[0], v.extrinsic[1])
        # print(extrinsic,extrinsic.shape)
        
        ipm_img=fisheye_to_ipm(img,camera_model,extrinsic[0:3,:],ipm_width,ipm_height,scale)
        if 'BACK' in k or 'FRONT' in k:
            # print(cam_id)
            ipm_img_new=cv2.flip(ipm_img, 0)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ipm_imgs.append(ipm_img_new)
            # cv2.imwrite(f'training_validate/IPM/result_' +k +'.jpg', ipm_img_new)
        if 'LEFT'in k or 'RIGHT' in k:
            # print(cam_id)
            ipm_img_new=cv2.flip(ipm_img, 1)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_CLOCKWISE)
            ipm_imgs.append(ipm_img_new)
            # cv2.imwrite(f'training_validate/IPM/result_' +k +'.jpg', ipm_img_new)

    ipm_mask_dict = generate_camera_fusion_mask(1280, 24, angle=75, fusion_angle=15)
    # ipm_imgs: front left rear right
    ipm_image = torch.zeros_like(torch.from_numpy(ipm_imgs[0]).permute(2, 0, 1).float())
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_rear']).unsqueeze(0) * torch.from_numpy(ipm_imgs[0]).permute(2, 0, 1).float()
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_left']).unsqueeze(0) * torch.from_numpy(ipm_imgs[1]).permute(2, 0, 1).float()   
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_front']).unsqueeze(0) * torch.from_numpy(ipm_imgs[2]).permute(2, 0, 1).float()  
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_right']).unsqueeze(0) * torch.from_numpy(ipm_imgs[3]).permute(2, 0, 1).float()
    ipm_image = (torch.tensor(ipm_image).permute(1, 2, 0).numpy()).astype('uint8')



    #在IPM图像上画原始gt停车位
    center_x, center_y = ipm_width/2, ipm_height/2  # 图像中心
    overlay = ipm_image.copy()
    clean_ipm = ipm_image.copy()  # 保存没有任何overlay的原图
    # 半透明绿色（BGR = (0, 255, 0)）
    line_color = (0, 255, 0)
    alpha = 0.2  # 透明度：0 = 完全透明，1 = 不透明
    for parking_slot in batched_input_dict['transformables'][0]['parkingslot_3d'].elements:
        poly = parking_slot['points']
        world_xy= poly[:, :2]
        img_xy = np.zeros_like(world_xy, dtype=int)
        img_xy[:, 0] = (center_x - world_xy[:, 1] / 0.01875).astype(int)
        img_xy[:, 1] = (center_y - world_xy[:, 0] / 0.01875).astype(int)
        # 画点（不透明）
        for i, (u, v) in enumerate(img_xy):
            cv2.circle(ipm_image, (u, v), 5, (128, 0, 0), -1)
            if i==3 or i==0:
                cv2.putText(ipm_image, str(i), (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # 在 overlay 图层上画线（透明图层上画）
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[1]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[1]), tuple(img_xy[2]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[3]), line_color, 2)
    # 混合透明图层到原图
    cv2.addWeighted(overlay, alpha, ipm_image, 1 - alpha, 0, dst=ipm_image)
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id
        if not os.path.exists('training_validate/IPM_gt'):
            os.makedirs('training_validate/IPM_gt')
        cv2.imwrite(f'training_validate/IPM_gt/result_{scene_id}${frame_id}.jpg',cv2.cvtColor(ipm_image, cv2.COLOR_RGB2BGR))

    #在IPM图像上画heatmap ground truth
    gt_parking = {**gt}
    start_x = (1280 - 960) // 2         
    end_x = start_x + 960       
    cropped_ipm_image = clean_ipm[:, start_x:end_x, :]  
    # pdb.set_trace()
    im_save = torch.from_numpy(cropped_ipm_image.astype(np.float32)).permute(2, 0, 1)
    im_save = im_save.unsqueeze(0)
    hm_pts= gt_parking['pts']
    hm_pin= gt_parking['pin']
    # print(hm_pin.max())
    heatmap = torch.cat([hm_pts, hm_pin], dim=0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(640,480), mode='bilinear', align_corners=False)
    
    pred_batch_img = batch_with_heatmap(im_save/255, heatmap) 
    pred_batch_img = cv2.cvtColor(pred_batch_img, cv2.COLOR_BGR2RGB)
    # pdb.set_trace()
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id
        if not os.path.exists('training_validate/hm_gt'):
            os.makedirs('training_validate/hm_gt')
        cv2.imwrite(f'training_validate/hm_gt/result_{scene_id}${frame_id}.jpg',pred_batch_img)


    #在IPM图像上画tensorsmith reverse停车位
    center_x, center_y = ipm_width/2, ipm_height/2  # 图像中心
    overlay = ipm_image.copy()
    # 半透明绿色（BGR = (0, 255, 0)）
    line_color = (0, 255, 0)
    alpha = 0.2  # 透明度：0 = 完全透明，1 = 不透明
    for box in results[0]:
        world_xy = box[:, :2]
        img_xy = np.zeros_like(world_xy, dtype=int)
        img_xy[:, 0] = (center_x - world_xy[:, 1] / 0.01875).astype(int)
        img_xy[:, 1] = (center_y - world_xy[:, 0] / 0.01875).astype(int)
        # 画点（不透明）
        for i, (u, v) in enumerate(img_xy):
            cv2.circle(ipm_image, (u, v), 5, (128, 0, 0), -1)
            if i==3 or i==0:
                cv2.putText(ipm_image, str(i), (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # 在 overlay 图层上画线（透明图层上画）
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[1]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[1]), tuple(img_xy[2]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[3]), line_color, 2)
    # 混合透明图层到原图
    cv2.addWeighted(overlay, alpha, ipm_image, 1 - alpha, 0, dst=ipm_image)
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id
        if not os.path.exists('training_validate/IPM_reverse'):
            os.makedirs('training_validate/IPM_reverse')
        cv2.imwrite(f'training_validate/IPM_reverse/result_{scene_id}${frame_id}.jpg',cv2.cvtColor(ipm_image, cv2.COLOR_RGB2BGR))


    # print(ipm_image.shape)
    # pdb.set_trace()


    return img


def draw_results_parking(pred_dict, batched_input_dict, save_im=True):
    if False:  # dbg,draw labels
        lidar_points = batched_input_dict['transformables'][0]['lidar_sweeps'].positions
        gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor

        branch = 'bbox_3d_heading'
        transformables = batched_input_dict['transformables'][0]
        tensor_smith = transformables[branch].tensor_smith

        pred_dict_branch = pred_dict[branch]
        pred_dict_branch_0 = {**pred_dict_branch}
        for key in pred_dict_branch_0:
            if key in ['cen', 'seg','pts']:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
            else:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
        results = tensor_smith.reverse(pred_dict_branch_0)
        corners_ego = np.array(
            [pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation']) for box in results])
        draw_points_with_label(lidar_points, corners_ego)

    # get the result image here
    result_dict = {}
    gt = batched_input_dict['transformables'][0]['parkingslot_3d'].tensor

    for cam, v in batched_input_dict['transformables'][0]['camera_images'].transformables.items():
        img0 = v.tensor['img']
        img = img0.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 128
        camera_model = get_fisheye_camera_model(v, cam)
        Tce = np.linalg.inv(Rt2T(v.extrinsic[0], v.extrinsic[1]))
        result_dict[cam] = (img, camera_model, Tce)

    branch = 'parkingslot_3d'
    # gt
    # transformables = batched_input_dict['transformables'][0]
    # tensor_smith = transformables[branch].tensor_smith
    # voxel_range = tensor_smith.voxel_range
    # results = tensor_smith.reverse(gt)
    # # draw results using tensorsmith reverse 
    # for k, v in result_dict.items():
    #     img, camera_model, Tce = result_dict[k]
    #     img_point1=copy.deepcopy(img)
    #     img_point2=copy.deepcopy(img)
    #     for poly in results[1]:
    #         corners_ego = poly        
    #         corners_ego_1x3=np.expand_dims(corners_ego[:3], axis=0).astype(np.float32)
    #         corners_2d = draw_camera_points_to_image_points(camera_model,
    #                                                         transform_pts_with_T(corners_ego_1x3, Tce))
    #         if corners_ego[4:5]==1:
    #             cv2.circle(img_point1, center=(int(round(corners_2d[0][0])), int(round(corners_2d[0][1]))), radius=3, color=(0, 0, 255), thickness=-1)
    #         if corners_ego[4:5]==2:
    #             cv2.circle(img_point2, center=(int(round(corners_2d[0][0])), int(round(corners_2d[0][1]))), radius=3, color=(255, 0, 0), thickness=-1)
        
    #         # from time import time
    #         if save_im:
    #             frame_id = batched_input_dict['index_infos'][0].frame_id
    #             cv2.imwrite(f'training_validate/fisheye/result_{k}_{frame_id}_1.jpg', img_point1)
    #             cv2.imwrite(f'training_validate/fisheye/result_{k}_{frame_id}_2.jpg', img_point2)



    # draw results using original 4D annotations
    for k, v in result_dict.items():
        img_bev, camera_model, Tce = result_dict[k]
        img_bev2=copy.deepcopy(img_bev)
        draw_flag=0
        for parking_slot in batched_input_dict['transformables'][0]['parkingslot_3d'].elements:
            poly = parking_slot['points']        
            # print(poly)
            # print(poly[:2, 3])
            # assert 0
            if poly[:,0].max()>9 or poly[:,0].min()<-9:
                continue
            if poly[:,1].max()>12 or poly[:,1].min()<-12:
                continue
            corners_2d = draw_camera_points_to_image_points(camera_model,
                                                            transform_pts_with_T(poly[:, :3], Tce))
            
            if corners_2d[0][0]<0 or corners_2d[0][1]<0 or corners_2d[0][0]>img_bev.shape[1] or corners_2d[0][1]>img_bev.shape[0]:
                continue
            # print(batched_input_dict['index_infos'][0].scene_id,batched_input_dict['index_infos'][0].frame_id,int(round(corners_2d[0][0])), int(round(corners_2d[0][1])))
            # assert 0
            cv2.circle(img_bev, center=(int(round(corners_2d[0][0])), int(round(corners_2d[0][1]))), radius=3, color=(0, 0, 255), thickness=-1)
            draw_flag=draw_flag+1
            cv2.circle(img_bev2, center=(int(round(corners_2d[1][0])), int(round(corners_2d[1][1]))), radius=3, color=(0, 255, 0), thickness=-1)
            # cv2.circle(img_bev, center=(int(round(corners_2d[2][0])), int(round(corners_2d[2][1]))), radius=3, color=(255, 0, 0), thickness=-1)
            # cv2.circle(img_bev, center=(int(round(corners_2d[3][0])), int(round(corners_2d[3][1]))), radius=3, color=(255, 255, 0), thickness=-1)
            # try:
            #     # draw_boxes(img, corners_2d)
            #     draw_polys(img, corners_2d)
            # except Exception:
            #     pass   
        # from time import time
        if save_im:
            frame_id = batched_input_dict['index_infos'][0].frame_id
            scene_id = batched_input_dict['index_infos'][0].scene_id
            if not os.path.exists('training_validate/fisheye_bev'):
                os.makedirs('training_validate/fisheye_bev')
            cv2.imwrite(f'training_validate/fisheye_bev/result_{scene_id}_{frame_id}_{k}_1.jpg', img_bev)
            cv2.imwrite(f'training_validate/fisheye_bev/result_{scene_id}_{frame_id}_{k}_2.jpg', img_bev2)   
    return img

def get_points_camera(points, camera, target_size):
    camera_points = camera.extrinsic[0].T @ (points.T - np.float32(camera.extrinsic[1])[None].T)
    uu, vv = camera.project_points_from_camera_to_image(camera_points)
    valid = (uu >= 0) & (uu < target_size[1]) & (vv >= 0) & (vv < target_size[0])
    return np.float32([uu[valid], vv[valid]]).T


def draw_3d_box_on_fisheye(img, bbox_3d, camera_model, Tce, target_size, color=(0, 255, 0), thickness=2, line_alpha=0.6):
    """
    Draw a 3D bounding box on fisheye image with dense sampling for smooth curves.

    Parameters
    ----------
    img : np.ndarray
        Fisheye image (will be modified in-place)
    bbox_3d : dict
        3D bounding box with keys: 'size', 'rotation', 'translation'
        - 'size': [l, w, h] in meters (or [radius, h] for cylinders)
        - 'rotation': 3x3 rotation matrix
        - 'translation': [x, y, z] center position in ego coordinates
    camera_model : FisheyeCameraModel
        Fisheye camera model for projection
    Tce : np.ndarray
        4x4 transformation matrix from ego to camera coordinates
    target_size : tuple
        (height, width) of target image
    color : tuple
        BGR color for drawing
    thickness : int
        Line thickness
    line_alpha : float
        Transparency for lines (0.0 = transparent, 1.0 = opaque), default 0.6

    Returns
    -------
    img : np.ndarray
        Image with 3D box drawn on it
    """
    # Extract bbox parameters
    translation = np.float32(bbox_3d['translation'])

    # Create overlay for transparent lines
    overlay = img.copy()

    # Handle different bbox formats from different tensor_smith types
    if 'size' in bbox_3d and 'rotation' in bbox_3d:
        # PlanarBbox3D format: has 'size' and 'rotation'
        size = np.float32(bbox_3d['size'])
        rotation = np.float32(bbox_3d['rotation']).reshape(3, 3)

        # Extract basis vectors
        xvec = 0.5 * rotation[:, 0] * size[0]
        yvec = 0.5 * rotation[:, 1] * size[1]
        zvec = 0.5 * rotation[:, 2] * size[2]

        # Define 12 edges of the 3D box with dense sampling (similar to demo_video)
        # Bottom face edges (z = -zvec)
        t = np.arange(0, 2.1, 0.1).reshape(-1, 1)  # Dense sampling
        line_segments = [
            translation + xvec - yvec - zvec + yvec * t,  # edge 0-1
            translation + xvec + yvec - zvec - xvec * t,  # edge 1-2
            translation - xvec + yvec - zvec - yvec * t,  # edge 2-3
            translation - xvec - yvec - zvec + xvec * t,  # edge 3-0
            # Vertical edges
            translation + xvec - yvec - zvec + zvec * t,  # edge 0-4
            translation + xvec + yvec - zvec + zvec * t,  # edge 1-5
            translation - xvec + yvec - zvec + zvec * t,  # edge 2-6
            translation - xvec - yvec - zvec + zvec * t,  # edge 3-7
            # Top face edges (z = +zvec)
            translation + xvec - yvec + zvec + yvec * t,  # edge 4-5
            translation + xvec + yvec + zvec - xvec * t,  # edge 5-6
            translation - xvec + yvec + zvec - yvec * t,  # edge 6-7
            translation - xvec - yvec + zvec + xvec * t,  # edge 7-4
        ]

        # Project and draw each edge on overlay
        for line in line_segments:
            # Transform to camera coordinates
            camera_points = transform_pts_with_T(line, Tce)
            # Project to image
            projected = camera_model.project_points(camera_points)

            # Filter valid points
            valid_pts = []
            for pt in projected:
                if pt is not None and len(pt) == 2:
                    u, v = pt
                    if 0 <= u < target_size[1] and 0 <= v < target_size[0]:
                        valid_pts.append((int(u), int(v)))

            # Draw polyline on overlay if we have valid points
            if len(valid_pts) >= 2:
                pts_array = np.array(valid_pts, dtype=np.int32)
                cv2.polylines(overlay, [pts_array], False, color, thickness)

        # Blend lines with transparency
        cv2.addWeighted(overlay, line_alpha, img, 1 - line_alpha, 0, img)

        # Draw front face with alpha if this is a heading object
        # Front face is composed of edges 0, 5, 8, 4
        front_face_lines = [line_segments[0], line_segments[5], line_segments[8][::-1], line_segments[4][::-1]]
        front_face_pts = []
        for line in front_face_lines:
            for pt_3d in line:
                camera_points = transform_pts_with_T(pt_3d.reshape(1, -1), Tce)
                projected = camera_model.project_points(camera_points)
                if projected[0] is not None and len(projected[0]) == 2:
                    u, v = projected[0]
                    if 0 <= u < target_size[1] and 0 <= v < target_size[0]:
                        front_face_pts.append((int(u), int(v)))

        # Fill front face with transparency
        if len(front_face_pts) >= 3:
            overlay_face = img.copy()
            pts_array = np.array(front_face_pts, dtype=np.int32)
            cv2.fillPoly(overlay_face, [pts_array], color)
            cv2.addWeighted(overlay_face, 0.3, img, 0.7, 0, img)

    elif 'radius' in bbox_3d and 'height' in bbox_3d and 'zvec' in bbox_3d:
        # PlanarCylinder3D format: has 'radius', 'height', 'zvec'
        radius = float(bbox_3d['radius'])
        height = float(bbox_3d['height'])
        zvec_raw = np.float32(bbox_3d['zvec'])

        # Get orthogonal basis
        zvec = zvec_raw / np.linalg.norm(zvec_raw)
        if abs(zvec[2]) < 0.9:
            xvec = np.cross(np.array([0, 0, 1], dtype=np.float32), zvec)
        else:
            xvec = np.cross(np.array([1, 0, 0], dtype=np.float32), zvec)
        xvec = xvec / np.linalg.norm(xvec)
        yvec = np.cross(zvec, xvec)
        yvec = yvec / np.linalg.norm(yvec)

        # Scale vectors
        zvec_scaled = 0.5 * height * zvec
        xvec_scaled = radius * xvec
        yvec_scaled = radius * yvec

        # Draw top and bottom circles
        round_angles = (np.arange(0, 360, 10) * np.pi / 180)[:, None]

        # Top circle
        top_points = translation + zvec_scaled + np.cos(round_angles) * xvec_scaled + np.sin(round_angles) * yvec_scaled
        camera_points = transform_pts_with_T(top_points, Tce)
        projected = camera_model.project_points(camera_points)
        valid_pts = []
        for pt in projected:
            if pt is not None and len(pt) == 2:
                u, v = pt
                if 0 <= u < target_size[1] and 0 <= v < target_size[0]:
                    valid_pts.append((int(u), int(v)))
        if len(valid_pts) >= 3:
            overlay_circle = img.copy()
            pts_array = np.array(valid_pts, dtype=np.int32)
            cv2.fillPoly(overlay_circle, [pts_array], color)
            cv2.addWeighted(overlay_circle, 0.5, img, 0.5, 0, img)

        # Bottom circle
        bottom_points = translation - zvec_scaled + np.cos(round_angles) * xvec_scaled + np.sin(round_angles) * yvec_scaled
        camera_points = transform_pts_with_T(bottom_points, Tce)
        projected = camera_model.project_points(camera_points)
        valid_pts = []
        for pt in projected:
            if pt is not None and len(pt) == 2:
                u, v = pt
                if 0 <= u < target_size[1] and 0 <= v < target_size[0]:
                    valid_pts.append((int(u), int(v)))
        if len(valid_pts) >= 3:
            overlay_circle = img.copy()
            pts_array = np.array(valid_pts, dtype=np.int32)
            cv2.fillPoly(overlay_circle, [pts_array], color)
            cv2.addWeighted(overlay_circle, 0.5, img, 0.5, 0, img)

        # Center line with transparency
        center_line = translation + zvec_scaled * np.arange(-1, 1.1, 0.1).reshape(-1, 1)
        camera_points = transform_pts_with_T(center_line, Tce)
        projected = camera_model.project_points(camera_points)
        valid_pts = []
        for pt in projected:
            if pt is not None and len(pt) == 2:
                u, v = pt
                if 0 <= u < target_size[1] and 0 <= v < target_size[0]:
                    valid_pts.append((int(u), int(v)))
        if len(valid_pts) >= 2:
            overlay_line = img.copy()
            pts_array = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(overlay_line, [pts_array], False, color, thickness + 1)
            cv2.addWeighted(overlay_line, line_alpha, img, 1 - line_alpha, 0, img)
    else:
        raise ValueError(f"Unsupported bbox_3d format. Expected either (size, rotation) or (radius, height, zvec), got keys: {bbox_3d.keys()}")

    return img


def get_circle_points_in_world(center_point, radius=0.05, num_points=16):
    """
    Generate circle points in world coordinates around a center point.

    Args:
        center_point: [x, y, z] center point in world coordinates
        radius: radius of the circle in meters (default 0.05m)
        num_points: number of points to generate for the circle

    Returns:
        numpy array of shape (num_points, 3) containing circle points
    """
    circle_points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center_point[0] + radius * np.cos(angle)
        y = center_point[1] + radius * np.sin(angle)
        z = center_point[2]
        circle_points.append([x, y, z])
    return np.array(circle_points)


def get_polyline_camera(polyline, camera_model, Tce, img_shape, interpolate_distance=0.1):
    """
    Project world coordinate polyline to camera image plane with interpolation.

    Args:
        polyline: list of [x, y, z] points in world coordinates
        camera_model: camera model for projection
        Tce: transformation matrix from ego to camera
        img_shape: (height, width) of the image
        interpolate_distance: interpolation distance in meters (default 0.1m)

    Returns:
        list of (u, v) pixel coordinates that are valid (within image bounds)
    """
    image_points = []

    for i in range(len(polyline) - 1):
        start_point = np.array(polyline[i])
        end_point = np.array(polyline[i + 1])

        # Calculate distance between points
        distance = np.linalg.norm(end_point - start_point)

        # Determine number of interpolation points
        if distance > interpolate_distance:
            num_points = int(distance / interpolate_distance) + 1
            # Interpolate points
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                interp_point = start_point + t * (end_point - start_point)

                # Transform to camera coordinates
                point_ego = np.array([interp_point[0], interp_point[1], interp_point[2], 1.0])
                point_cam = Tce @ point_ego

                # Project to image plane
                if point_cam[2] > 0:  # Check if point is in front of camera
                    try:
                        u, v = camera_model.project_points(point_cam[:3].reshape(1, -1))[0]
                        u_int, v_int = int(round(u)), int(round(v))
                        if 0 <= u_int < img_shape[1] and 0 <= v_int < img_shape[0]:
                            image_points.append((u_int, v_int))
                    except:
                        pass
        else:
            # For short segments, just project the start point
            point_ego = np.array([start_point[0], start_point[1], start_point[2], 1.0])
            point_cam = Tce @ point_ego
            if point_cam[2] > 0:
                try:
                    u, v = camera_model.project_points(point_cam[:3].reshape(1, -1))[0]
                    u_int, v_int = int(round(u)), int(round(v))
                    if 0 <= u_int < img_shape[1] and 0 <= v_int < img_shape[0]:
                        image_points.append((u_int, v_int))
                except:
                    pass

    # Always try to add the last point
    last_point = np.array(polyline[-1])
    point_ego = np.array([last_point[0], last_point[1], last_point[2], 1.0])
    point_cam = Tce @ point_ego
    if point_cam[2] > 0:
        try:
            u, v = camera_model.project_points(point_cam[:3].reshape(1, -1))[0]
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < img_shape[1] and 0 <= v_int < img_shape[0]:
                image_points.append((u_int, v_int))
        except:
            pass

    return image_points


def draw_results_3dboxes(pred_dict, batched_input_dict, save_im=True):
    # get the result image here

    ipm_width = 1280
    ipm_height = 1280
    scale = 0.01875  # 每像素表示的米数
    result_dict = {}
    gt = batched_input_dict['transformables'][0]['parkingslot_3d'].tensor
    # gt_lane = batched_input_dict['transformables'][0]['polyline_3d'].tensor
    # gt_car = batched_input_dict['transformables'][0]['polyline_3d_car'].tensor
    # gt_obstacle = batched_input_dict['transformables'][0]['polyline_3d_obstacle'].tensor


    for cam, v in batched_input_dict['transformables'][0]['camera_images'].transformables.items():
        img0 = v.tensor['img']
        img = img0.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 128
        camera_model = get_fisheye_camera_model(v, cam)
        Tce = np.linalg.inv(Rt2T(v.extrinsic[0], v.extrinsic[1]))
        result_dict[cam] = (img, camera_model, Tce, v)

    branch = 'parkingslot_3d'
    # gt
    transformables = batched_input_dict['transformables'][0]
    tensor_smith = transformables[branch].tensor_smith
    voxel_range = tensor_smith.voxel_range
    results = tensor_smith.reverse(gt)
    # # gt_lane
    # branch_lane = 'polyline_3d'
    # tensor_smith_lane = transformables[branch_lane].tensor_smith
    # results_lane = tensor_smith_lane.reverse(gt_lane)
    # # gt_car
    # branch_car = 'polyline_3d_car'
    # tensor_smith_car = transformables[branch_car].tensor_smith
    # results_car = tensor_smith_car.reverse(gt_car)
    # # gt_obstacle
    # branch_obstacle = 'polyline_3d_obstacle'
    # tensor_smith_obstacle = transformables[branch_obstacle].tensor_smith
    # results_obstacle = tensor_smith_obstacle.reverse(gt_obstacle)

    # ===== Define class name to color mapping (BGR format) =====
    CLASS_NAME_TO_COLOR = {
        # Heading objects
        'passenger_car': (64, 64, 255),
        'bus': (64, 64, 255),
        'truck': (64, 64, 255),
        'fire_engine': (128, 64, 64),
        'motorcycle': (64, 64, 128),
        'bicycle': (64, 64, 128),
        'tricycle': (64, 64, 128),
        'cleaning_cart': (64, 64, 128),
        'shopping_cart': (64, 64, 128),
        'stroller': (64, 64, 128),
        'scooter': (64, 64, 128),
        # Plane heading objects
        'arrow': (255, 255, 255),
        # No heading objects
        'wheel_stopper': (64, 255, 64),
        'speed_bump': (128, 128, 64),
        'water_filled_barrier': (64, 64, 64),
        'cement_pier': (64, 64, 64),
        'fire_box': (128, 64, 64),
        'distribution_box': (128, 64, 64),
        # Square objects
        'pillar_rectangle': (255, 128, 255),
        'parking_lock': (128, 128, 128),
        'parking_lock_locked': (0, 128, 128),
        'waste_bin': (64, 64, 64),
        # Cylinder objects
        'pillar_cylinder': (255, 128, 255),
        'cone': (255, 0, 0),
        'bollard': (64, 64, 64),
        'roadblock': (64, 64, 64),
        'stone_ball': (64, 64, 64),
        'crash_barrel': (64, 64, 64),
        'fire_hydrant': (128, 64, 64),
        'warning_triangle': (255, 0, 0),
        'charging_infra': (64, 64, 64),
        # Oriented cylinder objects
        'pedestrian': (255, 64, 64),
    }

    # ===== Reverse all bbox_3d branches from pred_dict =====
    bbox_branches = [
        'bbox_3d_heading',
        'bbox_3d_plane_heading',
        'bbox_3d_no_heading',
        'pin_3d_no_heading',
        'bbox_3d_square',
        'bbox_3d_cylinder',
        'bbox_3d_oriented_cylinder'
    ]

    bbox_results = {}
    for branch_name in bbox_branches:
        # Check if branch exists in both transformables (for tensor_smith) and pred_dict (for predictions)
        if branch_name in transformables and branch_name in pred_dict:
            tensor_smith_bbox = transformables[branch_name].tensor_smith
            transformable = transformables[branch_name]
            pred_bbox_tensor = batched_input_dict['transformables'][0][branch_name].tensor
            results_bbox = tensor_smith_bbox.reverse(pred_bbox_tensor)
            if results_bbox is not None and len(results_bbox) > 0:
                # Store results with transformable to get class names later
                bbox_results[branch_name] = (results_bbox, transformable)
                # print(f"{branch_name}: {len(results_bbox)} boxes detected")

    # ===== Helper function to get class name from confs =====
    def get_class_name_from_bbox(bbox_3d, transformable):
        """Get the class name from bbox confs array."""
        if 'confs' in bbox_3d and hasattr(transformable, 'dictionary'):
            confs = bbox_3d['confs']
            # confs[0] is the confidence for object presence
            # confs[1:1+num_classes] are class confidences
            dictionary = transformable.dictionary
            if 'classes' in dictionary and len(confs) > 1:
                num_classes = len(dictionary['classes'])
                class_confs = confs[1:1+num_classes]
                if len(class_confs) > 0:
                    class_idx = np.argmax(class_confs)
                    class_name = dictionary['classes'][class_idx]
                    # Extract short name from full class name
                    # e.g., 'class.vehicle.passenger_car' -> 'passenger_car'
                    if '.' in class_name:
                        class_name = class_name.split('.')[-1]
                    # Also check for '::' which is used for some classes
                    if '::' in class_name:
                        # e.g., 'class.traffic_facility.soft_barrier::attr.traffic_facility.soft_barrier.type.water_filled_barrier'
                        # Extract the attr part
                        parts = class_name.split('::')
                        if len(parts) > 1 and 'attr.' in parts[1]:
                            attr_name = parts[1].split('.')[-1]
                            return attr_name
                    return class_name
        return None

    # draw results using tensorsmith reverse

    # ===== First pass: Generate IPM without 3D boxes (for parking slots only visualization) =====
    ipm_imgs_no_boxes = []
    fisheye_imgs_no_boxes = {}  # Store clean fisheye images for slot visualization
    ipm_imgs_with_slots = []  # Store IPM images with parking slots drawn on fisheye first

    for k, v in result_dict.items():
        img, camera_model, Tce, v = result_dict[k]

        # Save clean fisheye image (without 3D boxes) for slot visualization
        fisheye_imgs_no_boxes[k] = img.copy()

        extrinsic = Rt2T(v.extrinsic[0], v.extrinsic[1])
        ipm_img = fisheye_to_ipm(img, camera_model, extrinsic[0:3,:], ipm_width, ipm_height, scale)

        if 'BACK' in k or 'FRONT' in k:
            ipm_img_new = cv2.flip(ipm_img, 0)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ipm_imgs_no_boxes.append(ipm_img_new)
        if 'LEFT' in k or 'RIGHT' in k:
            ipm_img_new = cv2.flip(ipm_img, 1)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_CLOCKWISE)
            ipm_imgs_no_boxes.append(ipm_img_new)

        # ===== Draw parking slots on fisheye image using helper functions, then convert to IPM =====
        img_with_slots = img.copy()
        img_height, img_width = img_with_slots.shape[:2]

        # Ensure img_with_slots is uint8 type
        if img_with_slots.dtype != np.uint8:
            img_with_slots = img_with_slots.astype(np.uint8)

        # Create overlay for transparent drawing
        overlay = np.zeros_like(img_with_slots, dtype=np.uint8)

        # DEBUG: Print parking slot count
        # num_slots = len(batched_input_dict['transformables'][0]['parkingslot_3d'].elements)
        # print(f"[DEBUG] Camera {k}: Processing {num_slots} parking slots")
        # circles_drawn = 0
        # lines_drawn = 0
        # total_points = 0
        # points_in_front = 0
        # points_projected_success = 0
        # points_in_bounds = 0
        # first_point_info_printed = False

        # slot_idx = 0
        for parking_slot in batched_input_dict['transformables'][0]['parkingslot_3d'].elements:
            poly = parking_slot['points']  # Shape: (N, 3) in world coordinates

            # Draw parking slot corners as circles (0.05m radius in world coordinates)
            for i, point_3d in enumerate(poly):
                # total_points += 1

                # DEBUG: Print first corner of first slot
                # if slot_idx == 0 and i == 0 and not first_point_info_printed:
                #     print(f"[DEBUG] Camera {k}: First slot, first corner world coords: {point_3d}")
                #     first_point_info_printed = True

                # Generate circle points in world coordinates
                circle_points = get_circle_points_in_world(point_3d, radius=0.05, num_points=16)

                # Project circle points to camera image
                circle_image_points = []
                for circle_pt in circle_points:
                    point_ego = np.array([circle_pt[0], circle_pt[1], circle_pt[2], 1.0])
                    point_cam = Tce @ point_ego

                    # DEBUG: Print transformation for first point
                    # if slot_idx == 0 and i == 0 and len(circle_image_points) == 0 and total_points == 1:
                    #     print(f"[DEBUG] Camera {k}: point_ego = {point_ego}")
                    #     print(f"[DEBUG] Camera {k}: point_cam = {point_cam}")
                    #     print(f"[DEBUG] Camera {k}: point_cam[2] = {point_cam[2]}, in front = {point_cam[2] > 0}")

                    if point_cam[2] > 0:  # Point is in front of camera
                        # points_in_front += 1
                        try:
                            u, v = camera_model.project_points(point_cam[:3].reshape(1, -1))[0]
                            # points_projected_success += 1

                            # DEBUG: Print projection result for first point
                            # if slot_idx == 0 and i == 0 and len(circle_image_points) == 0 and points_projected_success == 1:
                            #     print(f"[DEBUG] Camera {k}: Projected to u={u}, v={v}")
                            #     print(f"[DEBUG] Camera {k}: Image bounds: width={img_width}, height={img_height}")

                            u_int, v_int = int(round(u)), int(round(v))
                            if 0 <= u_int < img_width and 0 <= v_int < img_height:
                                # points_in_bounds += 1
                                circle_image_points.append((u_int, v_int))
                        except Exception as e:
                            # if total_points == 1:
                            #     print(f"[DEBUG] Camera {k}: Projection exception: {e}")
                            pass

                # Draw filled circle using polygon on overlay (red color for corners)
                if len(circle_image_points) >= 3:
                    pts_array = np.array(circle_image_points, dtype=np.int32)
                    cv2.fillPoly(overlay, [pts_array], (255, 0, 0))  # Red color for corners (RGB format)
                    # Add outline to make corners more visible
                    cv2.polylines(overlay, [pts_array], True, (255, 0, 0), 2)  # Red outline
                    # circles_drawn += 1

            # Draw connecting lines with interpolation
            # Connect corners: 0-1, 1-2, 0-3
            polyline_segments = [
                [poly[0], poly[1]],  # Edge 0-1
                [poly[1], poly[2]],  # Edge 1-2
                [poly[0], poly[3]]   # Edge 0-3
            ]

            for segment in polyline_segments:
                line_points = get_polyline_camera(segment, camera_model, Tce,
                                                   (img_height, img_width), interpolate_distance=0.1)
                # Draw the line on overlay (keep green color for lines)
                if len(line_points) >= 2:
                    pts_array = np.array(line_points, dtype=np.int32)
                    cv2.polylines(overlay, [pts_array], False, (0, 255, 0), 2)  # Green color for lines
                    # lines_drawn += 1

            # slot_idx += 1

        # Blend overlay with original image using alpha=0.5 (50% transparency)
        cv2.addWeighted(overlay, 0.1, img_with_slots, 1.0, 0, img_with_slots)

        # DEBUG: Print drawing statistics
        # print(f"[DEBUG] Camera {k}: Drew {circles_drawn} circles and {lines_drawn} lines")
        # print(f"[DEBUG] Camera {k}: Total corners: {total_points}, In front: {points_in_front}, Projected: {points_projected_success}, In bounds: {points_in_bounds}")

        # DEBUG: Save fisheye with slots before IPM conversion to check if slots are drawn
        # if save_im:
        #     frame_id = batched_input_dict['index_infos'][0].frame_id
        #     scene_id = batched_input_dict['index_infos'][0].scene_id
        #     debug_dir = f'training_validate/debug_fisheye_slots_before_ipm/result_{scene_id}'
        #     os.makedirs(debug_dir, exist_ok=True)
        #     cv2.imwrite(f'{debug_dir}/{frame_id}_{k}.jpg', cv2.cvtColor(img_with_slots, cv2.COLOR_RGB2BGR))

        # Convert fisheye with slots to IPM
        ipm_img_slots = fisheye_to_ipm(img_with_slots, camera_model, extrinsic[0:3,:],
                                       ipm_width, ipm_height, scale)

        if 'BACK' in k or 'FRONT' in k:
            ipm_img_slots_new = cv2.flip(ipm_img_slots, 0)
            ipm_img_slots_new = cv2.rotate(ipm_img_slots_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ipm_imgs_with_slots.append(ipm_img_slots_new)
        if 'LEFT' in k or 'RIGHT' in k:
            ipm_img_slots_new = cv2.flip(ipm_img_slots, 1)
            ipm_img_slots_new = cv2.rotate(ipm_img_slots_new, cv2.ROTATE_90_CLOCKWISE)
            ipm_imgs_with_slots.append(ipm_img_slots_new)

    # ===== Second pass: Draw 3D boxes on fisheye and generate IPM with 3D boxes =====
    ipm_imgs=[]
    for k, v in result_dict.items():
        img, camera_model, Tce, v = result_dict[k]

        # ===== Draw 3D boxes on fisheye image =====
        img_height, img_width = img.shape[:2]
        target_size = (img_height, img_width)

        for branch_name, (boxes_3d, transformable) in bbox_results.items():
            for bbox_3d in boxes_3d:
                # Get class name and corresponding color
                class_name = get_class_name_from_bbox(bbox_3d, transformable)
                color = CLASS_NAME_TO_COLOR.get(class_name, (128, 128, 128))  # Default gray if not found
                # Debug print to see what class names we're getting
                # print(f"Branch: {branch_name}, Class: {class_name}, Color: {color}, confs: {bbox_3d.get('confs', 'N/A')}")
                img = draw_3d_box_on_fisheye(img, bbox_3d, camera_model, Tce, target_size, color=color, thickness=2)

        # ===== Save fisheye image with 3D boxes =====
        if save_im:
            frame_id = batched_input_dict['index_infos'][0].frame_id
            scene_id = batched_input_dict['index_infos'][0].scene_id

            # Fix: Correct save path (remove duplicate scene_id in path)
            save_dir = f'training_validate/fisheye_3dboxes/result_{scene_id}'
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(f'{save_dir}/{frame_id}_{k}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Also save to second path format
            os.makedirs('training_validate/fisheye_3dboxes', exist_ok=True)
            cv2.imwrite(f'training_validate/fisheye_3dboxes/result_{scene_id}${frame_id}_{k}.jpg',
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        extrinsic=Rt2T(v.extrinsic[0], v.extrinsic[1])
        # print(extrinsic,extrinsic.shape)

        ipm_img=fisheye_to_ipm(img,camera_model,extrinsic[0:3,:],ipm_width,ipm_height,scale)
        # print("k:", k)
        if 'BACK' in k or 'FRONT' in k:
            # print(cam_id)
            ipm_img_new=cv2.flip(ipm_img, 0)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ipm_imgs.append(ipm_img_new)
            # cv2.imwrite(f'training_validate/IPM/result_' +k +'.jpg', ipm_img_new)
        if 'LEFT'in k or 'RIGHT' in k:
            # print(cam_id)
            ipm_img_new=cv2.flip(ipm_img, 1)
            ipm_img_new = cv2.rotate(ipm_img_new, cv2.ROTATE_90_CLOCKWISE)
            ipm_imgs.append(ipm_img_new)
            # cv2.imwrite(f'training_validate/IPM/result_' +k +'.jpg', ipm_img_new)

    # ===== Save fisheye images with only parking slots drawn =====
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id

        for k, img_clean in fisheye_imgs_no_boxes.items():
            img_with_slots = img_clean.copy()
            camera_model = get_fisheye_camera_model(result_dict[k][3], k)
            Tce = result_dict[k][2]

            # Draw parking slot corners on fisheye image
            for parking_slot in batched_input_dict['transformables'][0]['parkingslot_3d'].elements:
                poly = parking_slot['points']  # Shape: (N, 3) in world coordinates

                # Project each corner point to fisheye image
                for i, point_3d in enumerate(poly):
                    # Transform from ego to camera coordinates
                    point_ego = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
                    point_cam = Tce @ point_ego

                    # Project to fisheye image
                    if point_cam[2] > 0:  # Point is in front of camera
                        # u, v = camera_model.project_points(point_cam[:3])
                        # print("point_cam[:3]", point_cam[:3])
                        # print("point_cam[:3].reshape(1, -1)", point_cam[:3].reshape(1, -1))
                        u, v = camera_model.project_points(point_cam[:3].reshape(1, -1))[0]
                        u_int, v_int = int(round(u)), int(round(v))

                        # Check if point is within image bounds
                        if 0 <= u_int < img_with_slots.shape[1] and 0 <= v_int < img_with_slots.shape[0]:
                            # Draw corner point
                            cv2.circle(img_with_slots, (u_int, v_int), 3, (0, 255, 0), -1)

                            # Draw corner index for corners 0 and 3
                            if i == 0 or i == 3:
                                cv2.putText(img_with_slots, str(i), (u_int + 5, v_int - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save to both path formats
            save_dir_slots = f'training_validate/fisheye_slots/result_{scene_id}'
            os.makedirs(save_dir_slots, exist_ok=True)
            cv2.imwrite(f'{save_dir_slots}/{frame_id}_{k}.jpg', cv2.cvtColor(img_with_slots, cv2.COLOR_RGB2BGR))

            os.makedirs('training_validate/fisheye_slots', exist_ok=True)
            cv2.imwrite(f'training_validate/fisheye_slots/result_{scene_id}${frame_id}_{k}.jpg',
                       cv2.cvtColor(img_with_slots, cv2.COLOR_RGB2BGR))

    # ===== Generate fused IPM images (with and without 3D boxes) =====
    ipm_mask_dict = generate_camera_fusion_mask(1280, 24, angle=75, fusion_angle=15)

    # IPM without 3D boxes (for parking slots visualization)
    ipm_image_no_boxes = torch.zeros_like(torch.from_numpy(ipm_imgs_no_boxes[0]).permute(2, 0, 1).float())
    ipm_image_no_boxes += torch.tensor(ipm_mask_dict['camera_sv_rear']).unsqueeze(0) * torch.from_numpy(ipm_imgs_no_boxes[0]).permute(2, 0, 1).float()
    ipm_image_no_boxes += torch.tensor(ipm_mask_dict['camera_sv_left']).unsqueeze(0) * torch.from_numpy(ipm_imgs_no_boxes[1]).permute(2, 0, 1).float()
    ipm_image_no_boxes += torch.tensor(ipm_mask_dict['camera_sv_front']).unsqueeze(0) * torch.from_numpy(ipm_imgs_no_boxes[2]).permute(2, 0, 1).float()
    ipm_image_no_boxes += torch.tensor(ipm_mask_dict['camera_sv_right']).unsqueeze(0) * torch.from_numpy(ipm_imgs_no_boxes[3]).permute(2, 0, 1).float()
    ipm_image_no_boxes = (torch.tensor(ipm_image_no_boxes).permute(1, 2, 0).numpy()).astype('uint8')

    # IPM with parking slots drawn on fisheye first (for slots-only visualization)
    ipm_image_with_slots = torch.zeros_like(torch.from_numpy(ipm_imgs_with_slots[0]).permute(2, 0, 1).float())
    ipm_image_with_slots += torch.tensor(ipm_mask_dict['camera_sv_rear']).unsqueeze(0) * torch.from_numpy(ipm_imgs_with_slots[0]).permute(2, 0, 1).float()
    ipm_image_with_slots += torch.tensor(ipm_mask_dict['camera_sv_left']).unsqueeze(0) * torch.from_numpy(ipm_imgs_with_slots[1]).permute(2, 0, 1).float()
    ipm_image_with_slots += torch.tensor(ipm_mask_dict['camera_sv_front']).unsqueeze(0) * torch.from_numpy(ipm_imgs_with_slots[2]).permute(2, 0, 1).float()
    ipm_image_with_slots += torch.tensor(ipm_mask_dict['camera_sv_right']).unsqueeze(0) * torch.from_numpy(ipm_imgs_with_slots[3]).permute(2, 0, 1).float()
    ipm_image_with_slots = (torch.tensor(ipm_image_with_slots).permute(1, 2, 0).numpy()).astype('uint8')

    # IPM with 3D boxes (current behavior)
    # ipm_imgs: front left rear right
    ipm_image = torch.zeros_like(torch.from_numpy(ipm_imgs[0]).permute(2, 0, 1).float())
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_rear']).unsqueeze(0) * torch.from_numpy(ipm_imgs[0]).permute(2, 0, 1).float()
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_left']).unsqueeze(0) * torch.from_numpy(ipm_imgs[1]).permute(2, 0, 1).float()
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_front']).unsqueeze(0) * torch.from_numpy(ipm_imgs[2]).permute(2, 0, 1).float()
    ipm_image += torch.tensor(ipm_mask_dict['camera_sv_right']).unsqueeze(0) * torch.from_numpy(ipm_imgs[3]).permute(2, 0, 1).float()
    ipm_image = (torch.tensor(ipm_image).permute(1, 2, 0).numpy()).astype('uint8')
    ipm_image1 = ipm_image.copy()

    # ===== Save IPM with only 3D boxes (before drawing parking slots) =====
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id

        # Save IPM with 3D boxes only
        os.makedirs('training_validate/3D_boxes_reverse_2d2', exist_ok=True)
        cv2.imwrite(f'training_validate/3D_boxes_reverse_2d2/result_{scene_id}${frame_id}.jpg',
                    cv2.cvtColor(ipm_image.copy(), cv2.COLOR_RGB2BGR))

        save_dir_3dboxes = f'training_validate/3D_boxes_reverse_2d/result_{scene_id}'
        os.makedirs(save_dir_3dboxes, exist_ok=True)
        cv2.imwrite(f'{save_dir_3dboxes}/{frame_id}.jpg', cv2.cvtColor(ipm_image.copy(), cv2.COLOR_RGB2BGR))


    #在IPM图像上画原始gt停车位
    center_x, center_y = ipm_width/2, ipm_height/2  # 图像中心
    overlay = ipm_image.copy()
    clean_ipm = ipm_image.copy()  # 保存没有任何overlay的原图
    # 半透明绿色（BGR = (0, 255, 0)）
    line_color = (0, 255, 0)
    alpha = 0.2  # 透明度：0 = 完全透明，1 = 不透明
    for parking_slot in batched_input_dict['transformables'][0]['parkingslot_3d'].elements:
        poly = parking_slot['points']
        world_xy= poly[:, :2]

        img_xy = np.zeros_like(world_xy, dtype=int)
        img_xy[:, 0] = (center_x - world_xy[:, 1] / 0.01875).astype(int)
        img_xy[:, 1] = (center_y - world_xy[:, 0] / 0.01875).astype(int)
        # 画点（不透明）
        for i, (u, v) in enumerate(img_xy):
            cv2.circle(ipm_image, (u, v), 5, (128, 0, 0), -1)
            if i==3 or i==0:
                cv2.putText(ipm_image, str(i), (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # 在 overlay 图层上画线（透明图层上画）
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[1]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[1]), tuple(img_xy[2]), line_color, 2)
        cv2.line(overlay, tuple(img_xy[0]), tuple(img_xy[3]), line_color, 2)
    # 混合透明图层到原图
    cv2.addWeighted(overlay, alpha, ipm_image, 1 - alpha, 0, dst=ipm_image)

    # ===== Save IPM with parking slots (drawn on fisheye first, then converted to IPM) =====
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id

        # Use the pre-generated IPM image with parking slots (drawn on fisheye, then converted)
        # This replaces the old approach of drawing slots directly on IPM

        # Path 1: training_validate/IPM_reverse_2d2/
        os.makedirs('training_validate/IPM_reverse_2d2', exist_ok=True)
        cv2.imwrite(f'training_validate/IPM_reverse_2d2/result_{scene_id}${frame_id}.jpg',
                    cv2.cvtColor(ipm_image_with_slots, cv2.COLOR_RGB2BGR))

        # Path 2: training_validate/IPM_reverse_2d/
        save_dir_slots = f'training_validate/IPM_reverse_2d/result_{scene_id}'
        os.makedirs(save_dir_slots, exist_ok=True)
        cv2.imwrite(f'{save_dir_slots}/{frame_id}.jpg', cv2.cvtColor(ipm_image_with_slots, cv2.COLOR_RGB2BGR))

    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id
        # if not os.path.exists('training_validate/IPM_gt'):
        #     os.makedirs('training_validate/IPM_gt')
        save_dir = f'training_validate/IPM_gt/result_{scene_id}'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/{frame_id}.jpg', cv2.cvtColor(ipm_image, cv2.COLOR_RGB2BGR))

    #在IPM图像上画heatmap ground truth
    gt_parking = {**gt}
    start_x = (1280 - 960) // 2         
    end_x = start_x + 960       
    cropped_ipm_image = clean_ipm[:, start_x:end_x, :]  
    # pdb.set_trace()
    im_save = torch.from_numpy(cropped_ipm_image.astype(np.float32)).permute(2, 0, 1)
    im_save = im_save.unsqueeze(0)
    hm_pts= gt_parking['pts']
    hm_pin= gt_parking['pin']
    # print(hm_pin.max())
    heatmap = torch.cat([hm_pts, hm_pin], dim=0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=(640,480), mode='bilinear', align_corners=False)
    
    pred_batch_img = batch_with_heatmap(im_save/255, heatmap) 
    pred_batch_img = cv2.cvtColor(pred_batch_img, cv2.COLOR_BGR2RGB)
    # pdb.set_trace()
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        scene_id = batched_input_dict['index_infos'][0].scene_id
        # if not os.path.exists('training_validate/hm_gt'):
        #     os.makedirs('training_validate/hm_gt')
        save_dir = f'training_validate/hm_gt/result_{scene_id}'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/{frame_id}.jpg', pred_batch_img)

                    
    # # ===== draw predicted car lines =====
    # if 'polyline_3d_car' in pred_dict:
    #     # print("draw car lines")
    #     if results_car is not None:
    #         for cell_points in results_car:
    #             # print("cell_points:", cell_points)

    #             (world_cell_start, world_cell_end) = cell_points
    #             cell_start = (int(center_x - world_cell_start[1] / 0.01875), int(center_y - world_cell_start[0] / 0.01875))
    #             cell_end = (int(center_x - world_cell_end[1] / 0.01875), int(center_y - world_cell_end[0] / 0.01875))
    #             cv2.line(ipm_image1, cell_start, cell_end, (255, 0, 0), 1)  # 蓝色线段
    #             # cv2.circle(ipm_image, cell_start, 3, (255, 0, 0), -1)  # 蓝色起点
    #             # 用叉号标注起点
    #             size = 3  # 叉号的大小
    #             color = (0, 0, 255)  # 蓝色
    #             cv2.line(ipm_image1, (cell_start[0] - size, cell_start[1] - size), 
    #                     (cell_start[0] + size, cell_start[1] + size), color, 1)  # 左上到右下
    #             cv2.line(ipm_image1, (cell_start[0] - size, cell_start[1] + size), 
    #                     (cell_start[0] + size, cell_start[1] - size), color, 1)  # 左下到右上
                
    #             cv2.circle(ipm_image1, cell_end, 3, (0, 255, 0), -1)  # 绿色终点

    # # ===== draw predicted obstacle lines =====
    # if 'polyline_3d_obstacle' in pred_dict:
    #     if results_obstacle is not None:
    #         polylines_points, polylines_labels = results_obstacle
    #         for (cell_points, cell_label) in zip(polylines_points, polylines_labels):
    #             # print("cell_points, cell_label:", cell_points, cell_label)

    #             (world_cell_start, world_cell_end) = cell_points
    #             cell_start = (int(center_x - world_cell_start[1] / 0.01875), int(center_y - world_cell_start[0] / 0.01875))
    #             cell_end = (int(center_x - world_cell_end[1] / 0.01875), int(center_y - world_cell_end[0] / 0.01875))
    #             cv2.line(ipm_image1, cell_start, cell_end, (0, 0, 255), 1)  # 蓝色线段
    #             # cv2.circle(ipm_image, cell_start, 3, (255, 0, 0), -1)  # 蓝色起点
    #             # 计算线段中点
    #             mid_point = ((cell_start[0] + cell_end[0]) // 2, (cell_start[1] + cell_end[1]) // 2)
                
    #             # 在中点位置显示 cell_label
    #             cv2.putText(ipm_image1, str(cell_label), mid_point, 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 255), 1, cv2.LINE_AA)  # 黄色标签
                
    #             # 用叉号标注起点
    #             size = 3  # 叉号的大小
    #             color = (0, 0, 255)  # 蓝色
    #             cv2.line(ipm_image1, (cell_start[0] - size, cell_start[1] - size), 
    #                     (cell_start[0] + size, cell_start[1] + size), color, 1)  # 左上到右下
    #             cv2.line(ipm_image1, (cell_start[0] - size, cell_start[1] + size), 
    #                     (cell_start[0] + size, cell_start[1] - size), color, 1)  # 左下到右上
                
    #             cv2.circle(ipm_image1, cell_end, 3, (0, 255, 0), -1)  # 绿色终点

    # # ===== draw BEV grid (stride=1 -> 0.15 m) =====
    # grid_overlay = ipm_image1.copy()
    # grid_step = int(round(0.30 / scale))  # 16
    # grid_color = (180, 180, 180)
    # thickness = 1
    # line_type = cv2.LINE_AA

    # # 向右移动 4 个像素
    # offset_x = 4
    # offset_y = 4

    # # 垂直线
    # for x in range(offset_x, ipm_width, grid_step):  # Vertical lines
    #     for y in range(offset_y, ipm_height, grid_step):
    #         start_point = (x, min(y, ipm_height))
    #         end_point = (x, min(y + grid_step, ipm_height))
    #         cv2.line(grid_overlay, start_point, end_point, grid_color, thickness, line_type)

    # # 水平线
    # for y in range(offset_y, ipm_height, grid_step):  # Horizontal lines
    #     for x in range(offset_x, ipm_width, grid_step):
    #         start_point = (min(x, ipm_width), y)
    #         end_point = (min(x + grid_step, ipm_width), y)
    #         cv2.line(grid_overlay, start_point, end_point, grid_color, thickness, line_type)

    # # 混合透明图层到原图
    # cv2.addWeighted(grid_overlay, 0.2, ipm_image1, 0.8, 0, dst=ipm_image1)
    # # 计算像素长度（假设前面已经有了）
    # pix6 = int(round(6.0 / scale))   # 6 米对应像素
    # pix3 = int(round(3.0 / scale))   # 3 米对应像素

    # # 图像中心
    # cx, cy = int(center_x), int(center_y)

    # # “前左”方向的两个目标点
    # pt6 = (cx - pix6, cy - pix6)   # 6m,6m
    # pt3 = (cx - pix3, cy - pix3)   # 3m,3m
    # print("pt6:", pt6)
    # print("pt3:", pt3)
    # 黑色线条，宽度 2
    # cv2.line(ipm_image1, pt6, pt3, (0,0,0), 2)

    if save_im:
        # ===== Create combined image: IPM on left, 2x2 fisheye on right =====
        # Get fisheye images with 3D boxes drawn (already stored in result_dict)
        fisheye_imgs = {}
        for cam, (img, camera_model, Tce, v) in result_dict.items():
            fisheye_imgs[cam] = img.copy()

        # Determine camera order (assuming standard camera names)
        # Layout: Front-top-left, Rear-top-right, Left-bottom-left, Right-bottom-right
        cam_order = []
        for prefix in ['FRONT', 'BACK', 'LEFT', 'RIGHT']:
            for cam_name in fisheye_imgs.keys():
                if prefix in cam_name.upper():
                    cam_order.append(cam_name)
                    break

        # Get original fisheye image size (assuming all fisheye images have same size)
        if cam_order and cam_order[0] in fisheye_imgs:
            orig_fisheye_h, orig_fisheye_w = fisheye_imgs[cam_order[0]].shape[:2]
        else:
            orig_fisheye_h, orig_fisheye_w = 320, 640  # Default size

        # Create 2x2 grid with original fisheye size (no resizing)
        grid_h = orig_fisheye_h * 2
        grid_w = orig_fisheye_w * 2
        fisheye_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Fill the 2x2 grid with original size fisheye images
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (row, col)
        for idx, cam_name in enumerate(cam_order[:4]):  # Ensure max 4 cameras
            if cam_name in fisheye_imgs:
                img = fisheye_imgs[cam_name]
                # Place in grid without resizing
                row, col = positions[idx]
                y_start = row * orig_fisheye_h
                y_end = y_start + orig_fisheye_h
                x_start = col * orig_fisheye_w
                x_end = x_start + orig_fisheye_w

                # Ensure the image fits in the grid
                img_h, img_w = img.shape[:2]
                if img_h == orig_fisheye_h and img_w == orig_fisheye_w:
                    fisheye_grid[y_start:y_end, x_start:x_end] = img
                else:
                    # If size doesn't match, place at top-left corner with black padding
                    fisheye_grid[y_start:y_start+img_h, x_start:x_start+img_w] = img

                # Add camera label
                label_pos = (x_start + 10, y_start + 30)
                cv2.putText(fisheye_grid, cam_name, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine IPM (left) and fisheye grid (right)
        # If heights don't match, pad with black
        if ipm_height != grid_h:
            if ipm_height > grid_h:
                # Pad fisheye grid vertically
                pad_h = ipm_height - grid_h
                fisheye_grid = np.vstack([fisheye_grid, np.zeros((pad_h, grid_w, 3), dtype=np.uint8)])
            else:
                # Pad IPM vertically
                pad_h = grid_h - ipm_height
                ipm_image1 = np.vstack([ipm_image1, np.zeros((pad_h, ipm_width, 3), dtype=np.uint8)])

        combined_img = np.hstack([ipm_image1, fisheye_grid])

        # Save combined image
        save_dir_combined = f'training_validate/bbox_reverse_all/result_{scene_id}'
        os.makedirs(save_dir_combined, exist_ok=True)
        cv2.imwrite(f'{save_dir_combined}/{frame_id}.jpg', cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    # print(ipm_image.shape)
    # assert False


    return img










def draw_results_planar_lidar(pred_dict, batched_input_dict, save_im=True):
    if False:  # dbg,draw labels
        lidar_points = batched_input_dict['transformables'][0]['lidar_sweeps'].positions
        gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor

        branch = 'bbox_3d_heading'
        transformables = batched_input_dict['transformables'][0]
        tensor_smith = transformables[branch].tensor_smith

        pred_dict_branch = pred_dict[branch]
        pred_dict_branch_0 = {**pred_dict_branch}
        for key in pred_dict_branch_0:
            if key in ['cen', 'seg']:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
            else:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
        results = tensor_smith.reverse(pred_dict_branch_0)
        corners_ego = np.array(
            [pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation']) for box in results])
        draw_points_with_label(lidar_points, corners_ego)

    # get the result image here
    result_dict = {}
    # gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor
    for cam, v in batched_input_dict['transformables'][0]['camera_images'].transformables.items():
        img0 = v.tensor['img']
        img = img0.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 128
        camera_model = get_fisheye_camera_model(v, cam)
        Tce = np.linalg.inv(Rt2T(v.extrinsic[0], v.extrinsic[1]))
        result_dict[cam] = (img, camera_model, Tce)

    branch = 'bbox_3d_heading'
    # pred
    pred_dict_branch = pred_dict[branch]
    pred_dict_branch_0 = {**pred_dict_branch}
    for key in pred_dict_branch_0:
        if key in ['cen', 'seg']:
            pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
        else:
            pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
    # ----------l
    # gt
    transformables = batched_input_dict['transformables'][0]
    tensor_smith = transformables[branch].tensor_smith
    voxel_range = tensor_smith.voxel_range
    # gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor
    # results = tensor_smith.reverse(gt)
    print(pred_dict_branch_0['seg'][5].numpy().max())
    results = tensor_smith.reverse(pred_dict_branch_0)
    for box in results:
        corners_ego = pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation'])
        for k, v in result_dict.items():
            img, camera_model, Tce = result_dict[k]
            corners_2d = draw_camera_points_to_image_points(camera_model,
                                                            transform_pts_with_T(corners_ego, Tce))
            try:
                draw_boxes(img, corners_2d)
            except Exception:
                pass
            if k == "VCAMERA_FISHEYE_FRONT":
                return img
    # from time import time
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        cv2.imwrite(f'/tmp/1234/1/result_{cam}_{frame_id}.jpg', img)
    return img



def draw_polys(img, points):
    cv2.polylines(img, [points.astype('int')],1, (0, 0, 255), 2)
    cv2.polylines(img, [points.astype('int')[:2]],0, (128, 255, 0), 3)
def draw_results_ps_lidar(pred_dict, batched_input_dict, save_im=True):
    if False:  # dbg,draw labels
        lidar_points = batched_input_dict['transformables'][0]['lidar_sweeps'].positions
        gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor

        branch = 'bbox_3d_heading'
        transformables = batched_input_dict['transformables'][0]
        tensor_smith = transformables[branch].tensor_smith

        pred_dict_branch = pred_dict[branch]
        pred_dict_branch_0 = {**pred_dict_branch}
        for key in pred_dict_branch_0:
            if key in ['cen', 'seg']:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
            else:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
        results = tensor_smith.reverse(pred_dict_branch_0)
        corners_ego = np.array(
            [pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation']) for box in results])
        draw_points_with_label(lidar_points, corners_ego)

    # get the result image here
    result_dict = {}
    gt = batched_input_dict['transformables'][0]['parkingslot_3d'].tensor
    for cam, v in batched_input_dict['transformables'][0]['camera_images'].transformables.items():
        img0 = v.tensor['img']
        img = img0.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 128
        camera_model = get_fisheye_camera_model(v, cam)
        Tce = np.linalg.inv(Rt2T(v.extrinsic[0], v.extrinsic[1]))
        result_dict[cam] = (img, camera_model, Tce)

    branch = 'parkingslot_3d'
    # pred
    pred_dict_branch = pred_dict[branch]
    pred_dict_branch_0 = {**pred_dict_branch}
    for key in pred_dict_branch_0:
        if key in ['cen', 'seg']:
            pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
        else:
            pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
    # ----------l
    # gt
    transformables = batched_input_dict['transformables'][0]
    tensor_smith = transformables[branch].tensor_smith
    voxel_range = tensor_smith.voxel_range
    # gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor
    # results = tensor_smith.reverse(gt)
    results = tensor_smith.reverse(pred_dict_branch_0)
    for poly in results:
        corners_ego = poly
        if poly[:2, 3].mean() > 0.1:
            # corners_ego = pred_planar_box_to_3d_corners(box['size'], box['rotation'], box['translation'])
            for k, v in result_dict.items():

                img, camera_model, Tce = result_dict[k]
                corners_2d = draw_camera_points_to_image_points(camera_model,
                                                                transform_pts_with_T(corners_ego[:, :3], Tce))
                try:
                    # draw_boxes(img, corners_2d)
                    draw_polys(img, corners_2d)
                except Exception:
                    pass
    # from time import time
    if save_im:
        frame_id = batched_input_dict['index_infos'][0].frame_id
        cv2.imwrite(f'/tmp/1234/1/result_{cam}_{frame_id}.jpg', img)
    return img


