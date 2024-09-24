import os
from argparse import ArgumentParser
import open3d as o3d
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from pathlib import Path as P
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion
from mmdet3d.datasets import NuScenesDataset
import matplotlib.patches as patches
from scipy.spatial.transform import Rotation
import ipdb
import sys
import mmengine
from virtual_camera import FisheyeCamera
import pdb

def parse_list(arg):
    # 去掉输入字符串的空格，并去掉前后的方括号
    items = arg.strip('[]').split(',')
    # 将每个元素转换为整数
    return [int(item) for item in items]

def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('idx',
                        type=parse_list,
                        help='Index of the dataset to be visualized.')
    parser.add_argument('middleware_data', help='Path of the result json file.')
    # parser.add_argument('target_path',
    #                     help='Target path to save the visualization result.')
    parser.add_argument('--test',action="store_true",
                        help='Only for the visualization result.')

    args = parser.parse_args()
    return args


def get_ego_box(box_dict, ego2global_rotation, ego2global_translation):
    box = Box(
        box_dict['translation'],
        box_dict['size'],
        Quaternion(box_dict['rotation']),
    )
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    box.translate(trans)
    box.rotate(rot)
    box_xyz = np.array(box.center)
    box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
    box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
    box_velo = np.array(box.velocity[:2])
    return np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones),
        axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot

def rotate_points_xyz(points, r_matrix):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    r_matrix = r_matrix.reshape(1,3,3)
    points_rot = np.matmul(points[:, :, 0:3], r_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def get_corners(boxes3d):
    """ 平面x右y上 轴
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
        BEV ego坐标 x朝前 y朝左(IMU坐标系)  旋转按照这个 BEV 视角下 角度向右负 向左正
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading],
            (x, y, z) is the box center
    Returns:
    """
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)

    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),
                                      boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d

def get_corners_with_angles(boxes3d, R_matrix):
    template = (np.array((
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    )) / 2)
    # R_matrix = Rotation.from_euler("xyz",angles=Rotation.from_matrix(R_matrix).as_euler("XYZ", degrees=False), degrees=False).as_matrix()
    corners3d = np.tile(boxes3d[:, None, 3:6],
                        [1, 8, 1]) * template[None, :, :]
    corners3d = rotate_points_xyz(corners3d.reshape(-1, 8, 3),
                                      R_matrix).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d

def get_bev_lines(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(4)]


def get_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]],
                        [corners[st, 1], corners[ed, 1]]])
    return ret

def get_fisheye_3d_lines(corners):
    ret = []
    for st, ed in [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
                   [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        if corners[st, -1] > 0 and corners[ed, -1] > 0:
            ret.append([[corners[st, 0], corners[ed, 0]],
                        [corners[st, 1], corners[ed, 1]]])
    return ret

def interpolate(p1, p2, num):
    return np.linspace(p1, p2, num=num, endpoint=True)

def interpolate_points(points):
    edges = [
      (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
      (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
      (0, 4), (1, 5), (2, 6), (3, 7)   # 连接底面和顶面的边
    ]
    new_points = []
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        distance = np.linalg.norm(p2 - p1)
        num_points = int(distance / 0.1) + 1
        new_points.append(interpolate(p1, p2, num_points))

    return new_points

def get_cam_corners(corners, translation, rotation, cam_intrinsics, mv4d=False):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners

def get_corners_ego2cam(corners, translation, rotation, mv4d=False):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    
    return cam_corners

def intrinsics_matrix(intrinsic):
    cx, cy, fx, fy = intrinsic
    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def demo(
    idx,
    middleware_data,
    test=False,
    threshold=0.5,
    show_range=60,
    show_classes=[
        'car',
        'truck',
        'construction_vehicle',
        'bus',
        'trailer',
        'barrier',
        'motorcycle',
        'bicycle',
        'pedestrian',
        'traffic_cone',
    ],
):
    # Set cameras
    nus = False
    av2 = False
    mv_4d = True
    if "nus" in middleware_data:
        nus = True
        IMG_KEYS = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT'
        ]
    elif "av2" in middleware_data:
        av2 = True
        IMG_KEYS = [
        'ring_rear_left', 'ring_side_left', 'ring_front_left', 'ring_front_center', 'ring_front_right', 'ring_side_right', 'ring_rear_right'
        ]
    elif "mv_4d" in middleware_data:
        mv_4d = True
        IMG_KEYS = [
        'VCAMERA_FISHEYE_FRONT', 'VCAMERA_PERSPECTIVE_FRONT_LEFT', 'VCAMERA_PERSPECTIVE_BACK_LEFT', 'VCAMERA_FISHEYE_LEFT', 'VCAMERA_PERSPECTIVE_BACK', 'VCAMERA_FISHEYE_BACK', 
        'VCAMERA_PERSPECTIVE_FRONT_RIGHT', 'VCAMERA_PERSPECTIVE_BACK_RIGHT', 'VCAMERA_FISHEYE_RIGHT', 'VCAMERA_PERSPECTIVE_FRONT'
        ]
    if test:
        infos = mmengine.load('data/nuScenes/nuscenes_infos_test.pkl')
        print("pkl load from:  data/nuScenes/nuscenes_infos_test.pkl")
    else:
        # infos = mmengine.load('data/nuScenes/nuscenes_infos_mini_mv.pkl')
        infos = mmengine.load(middleware_data)
        print(f"pkl load from:  {middleware_data}")
    # assert idx < len(infos)
    # Get data from dataset
    scene_id = middleware_data.split('/')[-1][12:-4]
    calibration = infos[scene_id]['scene_info']['calibration']
    timestamps = np.array(list(infos[scene_id]['frame_info'].keys()))
    for timestamp in timestamps[idx].tolist():
        dump_root = P(f"./work_dirs/vis_data/")
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_file = dump_root / P(str(timestamp) + ".jpg")
        frame_info = infos[scene_id]['frame_info'][timestamp]
        lidar_path = frame_info['lidar_points']['lidar1']
        if nus:
            lidar_data = o3d.t.io.read_point_cloud(os.path.join('data/nuScenes', lidar_path))
        elif av2:
            lidar_data = o3d.t.io.read_point_cloud(os.path.join('data/av2/sensor', lidar_path))
        elif mv_4d:
            lidar_data = o3d.io.read_point_cloud(os.path.join('data/mv_4d_data', lidar_path))
        lidar_points_positions = np.asarray(lidar_data.points)  # N * 3
        lidar_points_intensity = np.zeros_like(lidar_points_positions[:, 0:1])  # N * 1
        lidar_points = np.concatenate([lidar_points_positions, lidar_points_intensity], axis=1)  # N * 4
        # Get point cloud
        pts = lidar_points.copy()
        # Get GT corners
        gt_corners = []
        gt_labels = []
        for i in range(len(frame_info['3d_boxes'])):
            # if map_name_from_general_to_detection[
            #         info['ann_infos'][i]['category_name']] in show_classes:
                box = np.array(frame_info['3d_boxes'][i]['translation'].reshape(-1).tolist() + frame_info['3d_boxes'][i]['size'] + [Quaternion(matrix=frame_info['3d_boxes'][i]['rotation']).yaw_pitch_roll[0]] + [0, 0])
                if np.linalg.norm(box[:2]) <= show_range:
                    corners = get_corners_with_angles(box[None], frame_info['3d_boxes'][i]['rotation'].T)[0]
                    gt_corners.append(corners)
                    gt_labels.append(frame_info['3d_boxes'][i]['class'])

        # Set figure size
        if nus:
            plt.figure(figsize=(24, 8))
            row = 3 

            for i, k in enumerate(IMG_KEYS):
                # Draw camera views
                fig_idx = i + 1 if i < row else i + 2
                plt.subplot(2, 4, fig_idx)

                # Set camera attributes
                plt.title(k)
                plt.axis('off')
                plt.xlim(0, 1600)
                plt.ylim(900, 0)

                img = mmcv.imread(
                    os.path.join('data/nuScenes', info['cam_infos'][k]['filename']))  # type: ignore
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw images
                plt.imshow(img)

                # Draw 3D gt
                for corners in gt_corners:
                    cam_corners = get_cam_corners(
                        corners,
                        info['cam_infos'][k]['calibrated_sensor']['translation'],  # type: ignore
                        info['cam_infos'][k]['calibrated_sensor']['rotation'],   # type: ignore
                        info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic'])  # type: ignore
                    lines = get_3d_lines(cam_corners)
                    for line in lines:
                        plt.plot(line[0],
                                line[1],
                                c=cm.get_cmap('tab10')(4)
                                )
                # for box in info['box_2d'][k]['box']:
                #     x1, y1, x2, y2 = box
                #     w = x2 - x1
                #     h = y2 - y1
                #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                #     plt.gca().add_patch(rect)
            # Draw BEV
            plt.subplot(1, 4, 4)

            # Set BEV attributes
            plt.title('LIDAR_TOP')
            plt.axis('equal')
            plt.xlim(-40, 40)
            plt.ylim(-40, 40)

            # Draw point cloud
            plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')
            # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
            # Draw BEV GT boxes
            for corners in gt_corners:
                lines = get_bev_lines(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')

            # Set legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(),
                    by_label.keys(),
                    loc='upper right',
                    framealpha=1)

            # Save figure
            plt.tight_layout(w_pad=0, h_pad=2)
            plt.savefig(dump_file)
            plt.show()
        elif av2:
            plt.figure(figsize=(24, 8))
            row = 4 

            for i, k in enumerate(IMG_KEYS):
                # Draw camera views
                fig_idx = i + 1 if i < row else i + 2
                plt.subplot(2, 5, fig_idx)

                # Set camera attributes
                plt.title(k)
                plt.axis('off')
                plt.xlim(0, 2048)
                plt.ylim(1550, 0)

                img = mmcv.imread(
                    os.path.join('data/av2/sensor', info['cam_infos'][k]['filename']))  # type: ignore
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw images
                plt.imshow(img)

                # Draw 3D gt
                for corners in gt_corners:
                    cam_corners = get_cam_corners(
                        corners,
                        info['cam_infos'][k]['calibrated_sensor']['translation'],  # type: ignore
                        info['cam_infos'][k]['calibrated_sensor']['rotation'],  # type: ignore
                        info['cam_infos'][k]['calibrated_sensor']['camera_intrinsic'])   # type: ignore
                    lines = get_3d_lines(cam_corners)
                    for line in lines:
                        plt.plot(line[0],
                                line[1],
                                c=cm.get_cmap('tab10')(4)
                                )
                # for box in info['box_2d'][k]['box']:
                #     x1, y1, x2, y2 = box
                #     w = x2 - x1
                #     h = y2 - y1
                #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                #     plt.gca().add_patch(rect)
            # Draw BEV
            plt.subplot(1, 5, 5)

            # Set BEV attributes
            plt.title('LIDAR_TOP')
            plt.axis('equal')
            plt.xlim(-40, 40)
            plt.ylim(-40, 40)

            # Draw point cloud
            plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')
            # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
            # Draw BEV GT boxes
            for corners in gt_corners:
                lines = get_bev_lines(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')

            # Set legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(),
                    by_label.keys(),
                    loc='upper right',
                    framealpha=1)

            # Save figure
            plt.tight_layout(w_pad=0, h_pad=2)
            plt.savefig(dump_file)
            plt.show()
        elif mv_4d:
            plt.figure(figsize=(24, 8))
            row = 5 

            for i, k in enumerate(IMG_KEYS):
                # Draw camera views
                fig_idx = i + 1 if i < row else i + 2
                plt.subplot(2, 6, fig_idx)

                # Set camera attributes
                plt.title(k)
                plt.axis('off')
                if "FISHEYE" in k:
                    plt.xlim(0, 1024)
                    plt.ylim(640, 0)
                else:
                    plt.xlim(0, 1280)
                    plt.ylim(960, 0)
                img = mmcv.imread(
                    os.path.join('data/mv_4d_data', frame_info['camera_image'][k]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw images
                plt.imshow(img)

                # Draw 3D gt
                if "FISHEYE" in k:
                    for corners in gt_corners:
                        cam_corners = get_corners_ego2cam(
                            corners,
                            calibration[k]["extrinsic"][1],
                            Quaternion(matrix=calibration[k]["extrinsic"][0]),
                        )
                        # 12条线段
                        inter_cam_corners = interpolate_points(cam_corners)
                        for inter_cam_corner in inter_cam_corners:
                            # pdb.set_trace()
                            img_corners = FisheyeCamera(resolution=img.shape[:2][::-1], extrinsic=calibration[k]["extrinsic"], intrinsic=calibration[k]['intrinsic'], fov=225).project_points_from_camera_to_image(inter_cam_corner.T)
                            img_corners = np.stack([img_corners[0], img_corners[1]], axis=1)
                            mask = np.logical_or(img_corners[:, 0] <= 0, img_corners[:, 1] <= 0)
                            img_corners = img_corners[~mask, :]
                            plt.plot(img_corners[:, 0],
                                    img_corners[:, 1],
                                    c=cm.get_cmap('tab10')(4)
                                    )
                else:
                    for corners, label_name in zip(gt_corners, gt_labels):
                        cam_corners = get_cam_corners(
                            corners,
                            calibration[k]["extrinsic"][1],
                            Quaternion(matrix=calibration[k]["extrinsic"][0]),
                            intrinsics_matrix(calibration[k]['intrinsic'][:4]))
                        lines = get_3d_lines(cam_corners)
                        for line in lines:
                            plt.plot(line[0],
                                    line[1],
                                    c=cm.get_cmap('tab10')(4)
                                    )
                        # debug
                        # if "VCAMERA_PERSPECTIVE_FRONT" == k:
                        #     plt.text(cam_corners[0][0], cam_corners[0][1], label_name, fontsize=5, ha='center', va='bottom')
                # for box in info['box_2d'][k]['box']:
                #     x1, y1, x2, y2 = box
                #     w = x2 - x1
                #     h = y2 - y1
                #     rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
                #     plt.gca().add_patch(rect)
            # Draw BEV
            plt.subplot(1, 6, 6)

            # Set BEV attributes
            plt.title('LIDAR_TOP')
            plt.axis('equal')
            plt.xlim(-40, 40)
            plt.ylim(-40, 40)

            # Draw point cloud
            plt.scatter(-pts[:, 1], pts[:, 0], s=0.01, c=pts[:, -1], cmap='gray')
            # BEV box ego 是x朝前 y朝左,  可视化出来到图上是x朝右，y朝前，对应到图上x=-y,y=x 
            # Draw BEV GT boxes
            for corners in gt_corners:
                lines = get_bev_lines(corners)
                for line in lines:
                    plt.plot([-x for x in line[1]],
                            line[0],
                            c='r',
                            label='ground truth')

            # Set legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(),
                    by_label.keys(),
                    loc='upper right',
                    framealpha=1)

            # Save figure
            plt.tight_layout(w_pad=0, h_pad=2)
            plt.savefig(dump_file)
            # plt.show()

if __name__ == '__main__':
    args = parse_args()
    demo(
        args.idx,
        args.middleware_data,
        args.test
    )
