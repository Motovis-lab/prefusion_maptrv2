import argparse
import math
from pathlib import Path
import pickle

import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from loguru import logger
from copious.io.fs import ensured_path
from copious.cv.geometry import Box3d, points3d_to_homo


class NotOnImageError(Exception):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-path", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-save-dir", type=ensured_path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


args = parse_arguments()

def main():
    with open(args.pickle_path, 'rb') as f:
        data = pickle.load(f)[args.scene_id]
    plot_bbox_bev(data, ensured_path(args.result_save_dir / "bbox_bev"))
    plot_bbox_2d(data, ensured_path(args.result_save_dir / "bbox_2d"))
    plot_polyline_bev(data, ensured_path(args.result_save_dir / "polyline_bev"))


def _draw_rect(p0, p1, p5, p4, linewidth=1, color='r', alpha=1):
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p1[0], p5[0]), (p1[1], p5[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p5[0], p4[0]), (p5[1], p4[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p4[0], p0[0]), (p4[1], p0[1]), linewidth=linewidth, color=color, alpha=alpha)


def _draw_axis(origin, xaxis, yaxis):
    plt.plot([origin[0], origin[0] + xaxis[0]], [origin[1], origin[1] + xaxis[1]], color="r", marker='.')
    plt.plot([origin[0], origin[0] + yaxis[0]], [origin[1], origin[1] + yaxis[1]], color="green", marker='.')
    plt.scatter([origin[0]], [origin[1]], color="black", marker='o')


def _draw_text(position, text, fontsize=6):
    plt.text(position[0], position[1], text, fontsize=fontsize, ha='center', va='center')


def _draw_3d_box(img, corners):
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for i, edge in enumerate(edges):
        # first 4 edges are the front face of the box, draw them with border lines
        # the rest 8 edges should be draw with thiner lines
        if i < 4:
            cv2.line(
                img,
                (int(corners[edge[0], 0]), int(corners[edge[0], 1])),
                (int(corners[edge[1], 0]), int(corners[edge[1], 1])),
                (0, 0, 255),
                2,
            )
        else:
            cv2.line(
                img,
                (int(corners[edge[0], 0]), int(corners[edge[0], 1])),
                (int(corners[edge[1], 0]), int(corners[edge[1], 1])),
                (0, 0, 255),
                1,
            )

def T4x4(rot3x3, translation):
    mat = np.eye(4)
    mat[:3, :3] = rot3x3
    mat[:3, 3] = translation.flatten()
    return mat

def K3x3(cx, cy, fx, fy):
    return np.array([[fx, 0,  cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])


def _plot_bbox_bev_of_single_frame(bbox_3d, ego_pose, save_path):
    ego2world = T4x4(ego_pose["rotation"], ego_pose["translation"])
    _ = plt.figure(figsize=(14, 14))
    _draw_axis(ego2world[:2, 3], *(ego2world[:2, :2].T * 2))  # scale := 2
    _draw_axis([0, 0], [1, 0], [0, 1])  # global axis
    for bbox in bbox_3d:
        _track_id = bbox['track_id']
        bbox = Box3d(bbox['translation'], np.array(bbox['size']), Rotation.from_matrix(bbox['rotation']))
        bbox_corners = points3d_to_homo(bbox.corners[[0, 1, 5, 4]]) @ ego2world.T
        _draw_rect(*bbox_corners.tolist(), color='blue', alpha=0.3)
        _draw_text(bbox_corners.mean(axis=0), _track_id)
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([-50, 50])
    plt.gca().set_ylim([-50, 50])
    plt.savefig(save_path)
    plt.close()


def plot_bbox_bev(data, save_dir):
    for frame_id, frame_info in tqdm(data['frame_info'].items()):
        _plot_bbox_bev_of_single_frame(frame_info["3d_boxes"], frame_info["ego_pose"], save_dir / f"{frame_id}.png")


def _plot_bbox_2d_of_single_frame(bbox_3d, calib, im_path, save_path):
    cam_extr, cam_intr = T4x4(*calib['extrinsic']), K3x3(*calib['intrinsic'])
    im = cv2.imread(im_path)
    h, w = im.shape[:2]
    
    for bbox in bbox_3d:
        bbox = Box3d(bbox['translation'], np.array(bbox['size']), Rotation.from_matrix(bbox['rotation']))
        bbox_corners = points3d_to_homo(bbox.corners)
        try:
            im_coords = pinhole_project(bbox_corners, cam_extr, cam_intr, (w, h))
        except NotOnImageError:
            continue
        _draw_3d_box(im, im_coords)

    cv2.imwrite(str(save_path), im)

def im_pts_within_image(pts, im_size):
    w, h = im_size
    return (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)

def pinhole_project(points_homo, cam_extr, cam_intr, im_size):
    T_cam_ego = np.linalg.inv(cam_extr)
    cam_coords = (T_cam_ego @ points_homo.T).T[:, :3]
    
    if (cam_coords[:, 2] < 0).any():
        raise NotOnImageError
    
    normalized_cam_coords = cam_coords[:, :2] / cam_coords[:, 2:3]
    im_coords = (cam_intr[:2, :2] @ normalized_cam_coords.T).T + cam_intr[:2, 2]
    
    if not im_pts_within_image(im_coords, im_size).all():
        raise NotOnImageError

    return im_coords


def plot_bbox_2d(data, save_dir):
    calib = data['scene_info']['calibration']
    for frame_id, frame_info in tqdm(data['frame_info'].items()):
        for cam_id, im_rel_path in frame_info['camera_image'].items():
            _plot_bbox_2d_of_single_frame(frame_info["3d_boxes"], calib[cam_id], args.data_root / im_rel_path, ensured_path(save_dir / cam_id) / f"{frame_id}.jpg")


def plot_polyline_bev(data, save_dir):
    pass


if __name__ == "__main__":
    main()
