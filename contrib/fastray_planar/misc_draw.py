import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtv4d import draw_points
from mtv4d.annos_4d.misc import boxes_to_corners_3d
from mtv4d.utils.box_base import to_corners_7, box_corners_to_dot_cloud
from mtv4d.utils.calib_base import transform_pts_with_T
from mtv4d.utils.draw_base import draw_boxes
from mtv4d.utils.sensors import FisheyeCameraModel  # same as copious

from prefusion import SegIouLoss, DualFocalLoss, PlanarBbox3D, PlanarRectangularCuboid, PlanarSquarePillar, \
    PlanarOrientedCylinder3D, PlanarCylinder3D, PlanarParkingSlot3D


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
    gt = batched_input_dict['transformables'][0]['bbox_3d_heading'].tensor
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


