# import sys
# sys.path.insert(0, "../../")

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import virtual_camera as vc

from scipy.io import loadmat
from pathlib import Path
from tqdm import tqdm
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

NAME2COLOR = {
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
    'arrow': (255, 255, 255),
    'wheel_stopper': (64, 255, 64),
    'speed_bump': (128, 128, 64),
    'water_filled_barrier': (64, 64, 64),
    'cement_pier': (64, 64, 64),
    'fire_box': (128, 64, 64),
    'distribution_box': (128, 64, 64),
    'pillar_rectangle': (255, 128, 255),
    'parking_lock': (128, 128, 128),
    'waste_bin': (64, 64, 64),
    'pillar_cylinder': (255, 128, 255),
    'cone': (255, 0, 0),
    'bollard': (64, 64, 64),
    'roadblock': (64, 64, 64),
    'stone_ball': (64, 64, 64),
    'crash_barrel': (64, 64, 64),
    'fire_hydrant': (128, 64, 64),
    'warning_triangle': (255, 0, 0),
    'charging_infra': (64, 64, 64),
    'pedestrian': (255, 64, 64),
}

heading_objs = [
    'pedestrian',
    'passenger_car',
    'bus',
    'truck',
    'fire_engine',
    'motorcycle',
    'bicycle',
    'tricycle',
    'cleaning_cart',
    'shopping_cart',
    'stroller',
    'scooter',
    'arrow'
]


# center ego
cameras = {
    'VCAMERA_FISHEYE_FRONT': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, -90), [2.43686209, 0.0055232 , 0.74317797]),
    'VCAMERA_FISHEYE_LEFT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 0), [0.71698729, 1.06778281, 1.01063169]),
    'VCAMERA_FISHEYE_BACK': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, 90), [-2.43845655, -1.53496593e-04,  1.00320036]),
    'VCAMERA_FISHEYE_RIGHT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 180), [ 0.75621105, -1.0802392 ,  1.02806573]),
}

# cameras_back = {
#     'VCAMERA_FISHEYE_FRONT': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, -90), [3.81686209, 0.0055232 , 0.74317797]),
#     'VCAMERA_FISHEYE_LEFT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 0), [2.09698729, 1.06778281, 1.01063169]),
#     'VCAMERA_FISHEYE_BACK': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, 90), [-1.05845655, -1.53496593e-04,  1.00320036]),
#     'VCAMERA_FISHEYE_RIGHT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 180), [ 2.13621105, -1.0802392 ,  1.02806573]),
# }

# cameras = {
#     'VCAMERA_FISHEYE_FRONT': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, -90), [2.24360394, -0.042407  ,  0.69432598]),
#     'VCAMERA_FISHEYE_LEFT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 0), [0.5738639831542967, 1.1372499465942385, 1.073140025138855]),
#     'VCAMERA_FISHEYE_BACK': vc.create_virtual_fisheye_camera((640, 384), (-120, 0, 90), [-2.419692039489746, 0.053640000522136154, 1.1092510223388672]),
#     'VCAMERA_FISHEYE_RIGHT': vc.create_virtual_fisheye_camera((640, 384), (-135, 0, 180), [0.47674098610878013, -1.0943230390548706, 1.080273985862732]),
# }

def get_box_bev(element):
    class_name = element['class']
    rotation = np.float32(element['rotation']).reshape(3, 3)
    translation = element['translation']
    size = element['size']
    xvec = 0.5 * rotation[:, 0] * size[0]
    yvec = 0.5 * rotation[:, 1] * size[1]
    corner_points = np.array([
        translation + xvec - yvec,
        translation + xvec + yvec,
        translation - xvec + yvec,
        translation - xvec - yvec
    ], dtype=np.float32)
    heading_points = np.array([
        corner_points[:2].mean(0),
        translation
    ], dtype=np.float32)
    return class_name, corner_points, heading_points
    

def get_box_camera(element, camera):
    class_name = element['class']
    rotation = np.float32(element['rotation']).reshape(3, 3)
    translation = element['translation']
    size = element['size']
    xvec = 0.5 * rotation[:, 0] * size[0]
    yvec = 0.5 * rotation[:, 1] * size[1]
    zvec = 0.5 * rotation[:, 2] * size[2]
    line_segments = np.float32([
        translation + xvec - yvec - zvec + yvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 01
        translation + xvec + yvec - zvec - xvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 12
        translation - xvec + yvec - zvec - yvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 23
        translation - xvec - yvec - zvec + xvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 30
        translation + xvec - yvec - zvec + zvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 04
        translation + xvec + yvec - zvec + zvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 15
        translation - xvec + yvec - zvec + zvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 26
        translation - xvec - yvec - zvec + zvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 37
        translation + xvec - yvec + zvec + yvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 45
        translation + xvec + yvec + zvec - xvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 56
        translation - xvec + yvec + zvec - yvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 67
        translation - xvec - yvec + zvec + xvec * np.arange(0, 2.1, 0.1).reshape(-1, 1),  # 74
    ])

    line_segments_camera = []
    for line in line_segments:
        camera_points = camera.extrinsic[0].T @ (line.T - np.float32(camera.extrinsic[1])[None].T)
        uu, vv = camera.project_points_from_camera_to_image(camera_points)
        valid = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 320)
        line_segments_camera.append(np.float32([uu[valid], vv[valid]]).T)

    front_face = np.concatenate([
        line_segments_camera[0],
        line_segments_camera[5],
        line_segments_camera[8][::-1],
        line_segments_camera[4][::-1]
    ])
    return class_name, line_segments_camera, front_face


def get_polyline_camera(polyline, camera):
    line_segments = []
    for i in range(len(polyline)-1):
        start = polyline[i]
        end = polyline[i+1]
        segment = np.array([start, end])
        # 计算线段长度
        segment_length = np.linalg.norm(end - start)
        # 如果长度超过0.5米则插值
        if segment_length > 0.5:
            direction = (end - start) / segment_length
            num_samples = int(np.ceil(segment_length / 0.5))
            t = np.linspace(0, segment_length, num_samples+1)
            interpolated = start + t[:, None] * direction
            line_segments.append(interpolated)
        else:
            line_segments.append(segment)
    line_segments_camera = []
    for line_segment in line_segments:
        camera_points = camera.extrinsic[0].T @ (line_segment.T - np.float32(camera.extrinsic[1])[None].T)
        uu, vv = camera.project_points_from_camera_to_image(camera_points)
        valid = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 320)
        if valid.sum() > 0:
            line_segments_camera.append(np.float32([uu[valid], vv[valid]]).T)

    return line_segments_camera


def get_points_camera(points, camera):
    camera_points = camera.extrinsic[0].T @ (points.T - np.float32(camera.extrinsic[1])[None].T)
    uu, vv = camera.project_points_from_camera_to_image(camera_points)
    valid = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 320)
    return np.float32([uu[valid], vv[valid]]).T


def get_cam_img(ind, info_gt, info_pred, info_lidar, cameras, results_root_path):
    # voxel_range = [(-1, 3), (12, -12), (9, -9)]
    # mat_path = Path(results_root_path / 'mat') / (ind + '.mat')
    # mat = loadmat(mat_path)
    # # 0.6225 ==> 0.5, 0.6682 ==> 0.7
    # occ_edge = cv2.resize(mat['occ_edge'], (480, 640)) > 0.6682
    # # height = cv2.resize(mat['height'], (480, 640))
    # hh, ww = occ_edge.nonzero()
    # xx = voxel_range[1][0] + hh * (voxel_range[1][1] - voxel_range[1][0]) / 640
    # yy = voxel_range[2][0] + ww * (voxel_range[2][1] - voxel_range[2][0]) / 480
    # zz = np.zeros_like(xx)
    # edge_points = np.float32([xx, yy, zz]).T

    imgs = []
    ori_imgs = []
    for cam_id in cameras:
        camera = cameras[cam_id]
        cam_path = Path(results_root_path / f'cameras/{cam_id}') / (ind + '.jpg')
        # cam_path = Path(cam_id) / (ind + '.jpg')
        img = plt.imread(cam_path)[:320]
        ori_imgs.append(img)
        plt.figure(figsize=(8, 4), dpi=80)
        fig = plt.gcf()
        plt.imshow(img)
        plt.axis('off')
        plt.axis([0, 639, 319, 0])
        
        # for slot in info['slots']:
        #     slot = np.float32(slot).reshape(-1, 4)[:, :3][[0, 1, 2, 3, 0]]
        #     slot_line_segments_camera = get_polyline_camera(slot, camera)
        #     for i, line in enumerate(slot_line_segments_camera):
        #         if i == 0:
        #             continue
        #         plt.plot(line[:, 0], line[:, 1], color=(0.6, 0.8, 0.6), linewidth=1)
        #     if len(slot_line_segments_camera) > 0:
        #         slot_area = np.concatenate(slot_line_segments_camera)
        #         plt.fill(slot_area[:, 0], slot_area[:, 1], color=(0.6, 0.8, 0.6), alpha=0.2)

        for element in info_gt['bboxes']:
            class_name, line_segments_camera, front_face = get_box_camera(element, camera)
            color = np.float32([64, 255, 64]) / 255
            for line in line_segments_camera:
                plt.plot(line[:, 0], line[:, 1], color=color)
            if class_name in heading_objs:
                plt.fill(front_face[:, 0], front_face[:, 1], color=color, alpha=0.3)

        for element in info_pred['bboxes']:
            # class_name, corner_points, heading_points = get_box_bev(element)
            class_name, line_segments_camera, front_face = get_box_camera(element, camera)
            color = np.float32(NAME2COLOR[class_name]) / 255
            for line in line_segments_camera:
                plt.plot(line[:, 0], line[:, 1], color=color)
            if class_name in heading_objs:
                plt.fill(front_face[:, 0], front_face[:, 1], color=color, alpha=0.3)
        
        for element in info_lidar['bboxes']:
            # class_name, corner_points, heading_points = get_box_bev(element)
            class_name, line_segments_camera, front_face = get_box_camera(element, camera)
            color = np.float32([255, 64, 64]) / 255
            for line in line_segments_camera:
                plt.plot(line[:, 0], line[:, 1], color=color)
            if class_name in heading_objs:
                plt.fill(front_face[:, 0], front_face[:, 1], color=color, alpha=0.3)

        # if 'polylines' in info:
        #     for polyline in info['polylines']:
        #         polyline = np.float32(polyline).reshape(-1, 3)
        #         line_segments_camera = get_polyline_camera(polyline, camera)
        #         for line in line_segments_camera:
        #             plt.plot(line[:, 0], line[:, 1], 'orange')
        
        # camera_edges = get_points_camera(edge_points, camera)
        # if len(camera_edges) > 0:
        #     plt.plot(camera_edges[:, 0], camera_edges[:, 1], '.', color='red', alpha=0.1)

        plt.tight_layout(pad=0)
        fig.canvas.draw()
        imgs.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close()
    img_cat = cv2.resize(np.concatenate(imgs, axis=0), (320, 640))
    ori_img_cat = cv2.resize(np.concatenate(ori_imgs, axis=0), (320, 640))
    # img_cat = cv2.resize(np.concatenate([
    #     np.concatenate(imgs[:2], axis=0), 
    #     np.concatenate(imgs[2:], axis=0)
    # ], axis=1), (1280, 640))
    return img_cat, ori_img_cat
    


def get_bev_img(info_gt, info_pred, info_lidar):
    voxel_range = [(-1, 3), (12, -12), (9, -9)]
    plt.figure(figsize=(6, 8), dpi=80)
    fig = plt.gcf()
    fig.set_facecolor((0.8, 0.8, 0.8))
    # fig.set_aspect('equal')
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    plt.axis('off')

    # mat_path = Path(results_root_path / 'mat') / (ind + '.mat')
    # mat = loadmat(mat_path)
    # # 0.6225 ==> 0.5, 0.6682 ==> 0.7
    # occ_edge = cv2.resize(mat['occ_edge'], (480, 640)) > 0.6682
    # # height = cv2.resize(mat['height'], (480, 640))
    # hh, ww = occ_edge.nonzero()
    # xx = voxel_range[1][0] + hh * (voxel_range[1][1] - voxel_range[1][0]) / 640
    # yy = voxel_range[2][0] + ww * (voxel_range[2][1] - voxel_range[2][0]) / 480
    # plt.plot(yy, xx, '.', color='red', alpha=0.02)
    # # plt.scatter(yy, xx, c=height[hh, ww], cmap='YlOrRd', vmin=0, vmax=2)


    # for slot in info['slots']:
    #     slot = np.float32(slot).reshape(-1, 4)[:, :3]
    #     plt.fill(slot[[0, 1, 2, 3, 0], 1], slot[[0, 1, 2, 3, 0], 0], 'gray', alpha=0.3)
    #     plt.plot(slot[[1, 2, 3, 0], 1], slot[[1, 2, 3, 0], 0], 'gray', linewidth=4)

    # if 'polylines' in info:
    #     for polyline in info['polylines']:
    #         polyline = np.float32(polyline).reshape(-1, 3)
    #         plt.plot(polyline[:, 1], polyline[:, 0], 'orange', linewidth=4)

    for element in info_gt['bboxes']:
        class_name, corner_points, heading_points = get_box_bev(element)
        color = np.float32([64, 255, 64]) / 255
        plt.fill(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color, alpha=0.5
        )
        plt.plot(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color
        )
        if class_name in heading_objs:
            plt.plot(heading_points[:, 1], heading_points[:, 0], color=color)

    for element in info_pred['bboxes']:
        class_name, corner_points, heading_points = get_box_bev(element)
        color = np.float32(NAME2COLOR[class_name]) / 255
        plt.fill(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color, alpha=0.5
        )
        plt.plot(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color
        )
        if class_name in heading_objs:
            plt.plot(heading_points[:, 1], heading_points[:, 0], color=color)
    
    
    for element in info_lidar['bboxes']:
        class_name, corner_points, heading_points = get_box_bev(element)
        color = np.float32([255, 64, 64]) / 255
        plt.fill(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color, alpha=0.5
        )
        plt.plot(
            corner_points[[0, 1, 2, 3, 0], 1], 
            corner_points[[0, 1, 2, 3, 0], 0], 
            color=color
        )
        if class_name in heading_objs:
            plt.plot(heading_points[:, 1], heading_points[:, 0], color=color)


    plt.tight_layout(pad=0)
    fig.canvas.draw()
    bev_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return bev_img


if __name__ == '__main__':
    results_root_path = Path('/home/alpha/Projects/PreFusion/work_dirs/borui_demo_dumps_20250304/gt_pred_dumps/')
    # all_inds = sorted([str(p.relative_to(results_root_path / 'dets'))[:-5] for p in results_root_path.glob('dets/**/*.json')])

    lidar_results_root_path = Path('/home/alpha/Projects/PreFusion/work_dirs/planar_lidar_dumps_0307/pred_dumps/')
    all_lidar_inds = sorted([str(p.relative_to(lidar_results_root_path / 'dets'))[:-5] for p in lidar_results_root_path.glob('dets/**/*.json')])

    demo_video = FFMPEG_VideoWriter('work_dirs/prefusion_liar_x.mp4', size=(320 * 2 + 480 * 1, 640), fps=10)
    for ind in tqdm(all_lidar_inds[::2]):
        json_path = results_root_path / 'dets' / (ind + '.json')
        json_gt_path = results_root_path / 'dets' / (ind + '_gt.json')
        lidar_json_path = lidar_results_root_path / 'dets' / (ind + '.json')

        info_pred = json.load(open(json_path))
        info_gt = json.load(open(json_gt_path))
        info_lidar = json.load(open(lidar_json_path))

        bev_img = get_bev_img(info_gt['gt'], info_pred['pred'], info_lidar['pred'])

        img_cat, ori_img_cat = get_cam_img(ind, info_gt['gt'], info_pred['pred'], info_lidar['pred'], cameras, results_root_path)

        # bev_img = cv2.resize(np.concatenate([bev_img_pred, bev_img_occ], axis=0), (240, 640))
        # img_final = np.concatenate([img_cat, bev_img], axis=1)
        img_final = np.concatenate([ori_img_cat, img_cat, bev_img], axis=1)
        # plt.imshow(img_final)
        # plt.show()
        # break

        demo_video.write_frame(img_final)
    demo_video.close()