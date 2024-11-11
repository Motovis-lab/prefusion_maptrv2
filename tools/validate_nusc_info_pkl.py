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
from copious.cv.camera_model import FisheyeCameraModel
from copious.io.parallelism import maybe_multiprocessing

from prefusion.dataset.utils import T4x4


class NotOnImageError(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-path", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-save-dir", type=ensured_path, required=True)
    parser.add_argument("--plot-range", nargs=4, type=float, default=(-50, 50, -50, 50), help="xlim[0], xlim[1], ylim[0], ylim[1]")
    parser.add_argument("--num-workers", type=int, default=0)
    return edict({k: v for k, v in parser.parse_args()._get_kwargs()})


args = parse_arguments()


def main():
    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)[args.scene_id]
    plot_bbox_bev(data, ensured_path(args.result_save_dir / "bbox_bev"))
    plot_bbox_2d(data, ensured_path(args.result_save_dir / "bbox_2d"))
    plot_bbox_velo(data, ensured_path(args.result_save_dir / "bbox_velo"))
    # plot_polyline_bev(data, ensured_path(args.result_save_dir / "polyline_bev"))
    # plot_polyline_2d(data, ensured_path(args.result_save_dir / "polyline_2d"))


def _draw_rect(p0, p1, p5, p4, linewidth=1, color="r", alpha=1):
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p1[0], p5[0]), (p1[1], p5[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p5[0], p4[0]), (p5[1], p4[1]), linewidth=linewidth, color=color, alpha=alpha)
    plt.plot((p4[0], p0[0]), (p4[1], p0[1]), linewidth=linewidth, color=color, alpha=alpha)


def _draw_polyline(vertices, linewidth=1, color="r", alpha=1):
    plt.plot(vertices[:, 0], vertices[:, 1], marker=".", linewidth=linewidth, color=color, alpha=alpha, markersize=2)


def _draw_direction(origin, direction, length=1, linewidth=1, color="k"):
    u = direction / np.linalg.norm(direction) * length
    plt.plot([origin[0], origin[0] + u[0]], [origin[1], origin[1] + u[1]], color=color, linewidth=linewidth)


def _draw_axis(origin, xaxis, yaxis):
    plt.plot([origin[0], origin[0] + xaxis[0]], [origin[1], origin[1] + xaxis[1]], color="r", marker=".", markersize=2)
    plt.plot([origin[0], origin[0] + yaxis[0]], [origin[1], origin[1] + yaxis[1]], color="green", marker=".", markersize=2)
    plt.scatter([origin[0]], [origin[1]], color="black", marker="o", s=3)


def _draw_text(position, text, fontsize=6):
    plt.text(position[0], position[1], text, fontsize=fontsize, ha="center", va="center")


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


def _draw_3d_polyline(img, vertices, color=(0, 255, 0)):
    if len(vertices) == 0:
        return
    next_vertex = vertices[-1]
    for i in range(len(vertices) - 1):
        cur_vertex, next_vertex = vertices[i], vertices[i + 1]
        cv2.circle(img, (int(cur_vertex[0]), int(cur_vertex[1])), 2, color, -1)
        cv2.line(
            img,
            (int(cur_vertex[0]), int(cur_vertex[1])),
            (int(next_vertex[0]), int(next_vertex[1])),
            color,
            2,
        )
    cv2.circle(img, (int(next_vertex[0]), int(next_vertex[1])), 2, color, -1)


def K3x3(cx, cy, fx, fy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def _plot_bbox_bev_of_single_frame(data_args):
    bbox_3d, ego_pose, save_path = data_args
    ego2world = T4x4(ego_pose["rotation"], ego_pose["translation"])
    _ = plt.figure()
    _draw_axis(ego2world[:2, 3], *(ego2world[:2, :2].T * 2))  # scale := 2
    _draw_axis([0, 0], [1, 0], [0, 1])  # global axis
    for bbox in bbox_3d:
        _track_id = bbox["track_id"]
        bbox = Box3d(bbox["translation"], np.array(bbox["size"]), Rotation.from_matrix(bbox["rotation"]))
        bbox_corners = points3d_to_homo(bbox.corners[[0, 1, 5, 4]]) @ ego2world.T
        _draw_rect(*bbox_corners.tolist(), color="blue", alpha=0.3)
        # _draw_text(bbox_corners.mean(axis=0), _track_id)
    plt.gca().set_aspect("equal")
    plt.gca().set_xlim([args.plot_range[0], args.plot_range[1]])
    plt.gca().set_ylim([args.plot_range[2], args.plot_range[3]])
    plt.savefig(save_path)
    plt.close()


def _plot_bbox_velo_of_single_frame(cur_boxes, cur_pose, next_boxes, next_pose, time_diff, save_path):
    cur_ego2world = T4x4(cur_pose["rotation"], cur_pose["translation"])
    next_ego2world = T4x4(next_pose["rotation"], next_pose["translation"])
    
    _ = plt.figure()
    
    _draw_axis(next_ego2world[:2, 3], *(next_ego2world[:2, :2].T * 2))  # scale := 2
    _draw_axis([0, 0], [1, 0], [0, 1])  # global axis

    for bbox in cur_boxes:
        cur_ego_velo = bbox["velocity"] * (time_diff / 1000)  # change time unit to second
        cur_world_velo = (cur_ego_velo[None, :2] @ cur_ego2world[:2, :2].T)[0]

        ori_box = Box3d(bbox["translation"], np.array(bbox["size"]), Rotation.from_matrix(bbox["rotation"]))
        ori_corners = points3d_to_homo(ori_box.corners[[0, 1, 5, 4]]) @ cur_ego2world.T
        
        predicted_box = Box3d(bbox["translation"] + cur_ego_velo, np.array(bbox["size"]), Rotation.from_matrix(bbox["rotation"]))
        predicted_corners = points3d_to_homo(predicted_box.corners[[0, 1, 5, 4]]) @ cur_ego2world.T
        
        _draw_rect(*ori_corners.tolist(), color="blue", alpha=0.2)
        _draw_rect(*predicted_corners.tolist(), color="green", alpha=0.2)
        
        if np.abs(cur_world_velo[:2]).sum() > 1e-2:
            _draw_direction(ori_corners.mean(axis=0)[:2], cur_world_velo[:2], length=0.3)

    for bbox in next_boxes:
        bbox = Box3d(bbox["translation"], np.array(bbox["size"]), Rotation.from_matrix(bbox["rotation"]))
        bbox_corners = points3d_to_homo(bbox.corners[[0, 1, 5, 4]]) @ next_ego2world.T
        _draw_rect(*bbox_corners.tolist(), color="red", alpha=0.2)
        
    plt.gca().set_aspect("equal")
    plt.gca().set_xlim([args.plot_range[0], args.plot_range[1]])
    plt.gca().set_ylim([args.plot_range[2], args.plot_range[3]])
    plt.savefig(save_path)
    plt.close()


def _plot_polyline_bev_of_single_frame(data_args):
    polylines, ego_pose, save_path = data_args
    ego2world = T4x4(ego_pose["rotation"], ego_pose["translation"])
    _ = plt.figure()
    _draw_axis(ego2world[:2, 3], *(ego2world[:2, :2].T * 2))  # scale := 2
    _draw_axis([0, 0], [1, 0], [0, 1])  # global axis
    for pl in polylines:
        _track_id = pl["track_id"]
        vertices = points3d_to_homo(pl["points"])
        vertices_world = vertices @ ego2world.T
        _draw_polyline(vertices_world, color="blue", alpha=0.3)
        # _draw_text(vertices_on_img[0], _track_id)
    plt.gca().set_aspect("equal")
    plt.gca().set_xlim([args.plot_range[0], args.plot_range[1]])
    plt.gca().set_ylim([args.plot_range[2], args.plot_range[3]])
    plt.savefig(save_path)
    plt.close()


def _plot_bbox_2d_of_single_frame(data_args):
    bbox_3d, calib, im_path, save_path = data_args
    im = cv2.imread(im_path)
    h, w = im.shape[:2]

    for bbox in bbox_3d:
        bbox = Box3d(bbox["translation"], np.array(bbox["size"]), Rotation.from_matrix(bbox["rotation"]))
        bbox_corners = points3d_to_homo(bbox.corners)
        try:
            im_coords = _project_points_to_image(bbox_corners, calib, (w, h))
        except NotOnImageError:
            continue
        _draw_3d_box(im, im_coords)

    cv2.imwrite(str(save_path), im)


def _plot_polyline_2d_of_single_frame(data_args):
    polylines, calib, im_path, save_path = data_args
    im = cv2.imread(im_path)
    h, w = im.shape[:2]

    for pl in polylines:
        vertices = points3d_to_homo(pl["points"])
        try:
            im_coords = _project_points_to_image(vertices, calib, (w, h))
        except NotOnImageError:
            continue
        _draw_3d_polyline(im, im_coords)

    cv2.imwrite(str(save_path), im)


def _project_points_to_image(points, calib, im_size):
    if calib["camera_type"] == "PerspectiveCamera":
        im_coords = pinhole_project(points, calib, im_size)
    elif calib["camera_type"] == "FisheyeCamera":
        im_coords = fisheye_project(points, calib, im_size)
    return im_coords


def im_pts_within_image(pts, im_size):
    w, h = im_size
    return (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)


def pinhole_project(points_homo, calib, im_size, conservative=True):
    cam_extr, cam_intr = T4x4(*calib["extrinsic"]), K3x3(*calib["intrinsic"][:4])
    T_cam_ego = np.linalg.inv(cam_extr)
    cam_coords = (T_cam_ego @ points_homo.T).T[:, :3]
    cam_coords = check_camera_coords_visibility_on_image(cam_coords, conservative)
    normalized_cam_coords = cam_coords[:, :2] / cam_coords[:, 2:3]
    im_coords = (cam_intr[:2, :2] @ normalized_cam_coords.T).T + cam_intr[:2, 2]
    im_coords = check_im_coords_visibility_on_image(im_coords, im_size, conservative)
    return im_coords


def fisheye_project(points_homo, calib, im_size, conservative=True):
    pp, focal, inv_poly = calib["intrinsic"][:2], calib["intrinsic"][2:4], calib["intrinsic"][4:]
    cam_extr = T4x4(*calib["extrinsic"])
    T_cam_ego = np.linalg.inv(cam_extr)
    cam_coords = (T_cam_ego @ points_homo.T).T[:, :3]
    cam_coords = check_camera_coords_visibility_on_image(cam_coords, conservative)
    cam_model = FisheyeCameraModel(pp, focal, inv_poly, im_size, 180)
    im_coords = cam_model.project_points(cam_coords)
    im_coords = check_im_coords_visibility_on_image(im_coords, im_size, conservative)
    return im_coords


def check_camera_coords_visibility_on_image(cam_coords, conservative=True):
    if conservative:
        if (cam_coords[:, 2] < 0).any():
            raise NotOnImageError
    else:
        if (cam_coords[:, 2] < 0).all():
            raise NotOnImageError
        cam_coords = cam_coords[cam_coords[:, 2] >= 0]
    return cam_coords


def check_im_coords_visibility_on_image(im_coords, im_size, conservative=True):
    if conservative:
        if not im_pts_within_image(im_coords, im_size).all():
            raise NotOnImageError
    else:
        if not im_pts_within_image(im_coords, im_size).any():
            raise NotOnImageError
        im_coords = im_coords[im_pts_within_image(im_coords, im_size)]
    return im_coords


def plot_bbox_bev(data, save_dir):
    data_args = [ (frame_info["3d_boxes"], frame_info["ego_pose"], save_dir / f"{frame_id}.png") for frame_id, frame_info in tqdm(data["frame_info"].items()) ]
    maybe_multiprocessing(_plot_bbox_bev_of_single_frame, data_args, num_processes=args.num_workers, use_tqdm=True, tqdm_desc="plotting bbox")


def plot_bbox_velo(data, save_dir):
    sorted_data = sorted(data["frame_info"].items(), key=lambda x: x[0])
    for i in tqdm(range(len(sorted_data) - 1)):
        cur_boxes = sorted_data[i][1]["3d_boxes"]
        cur_pose = sorted_data[i][1]["ego_pose"]
        next_boxes = sorted_data[i + 1][1]["3d_boxes"]
        next_pose = sorted_data[i + 1][1]["ego_pose"]
        time_diff = int(sorted_data[i + 1][0]) - int(sorted_data[i][0])
        save_path = save_dir / f"{sorted_data[i][0]}.png"
        _plot_bbox_velo_of_single_frame(cur_boxes, cur_pose, next_boxes, next_pose, time_diff, save_path)


def plot_bbox_2d(data, save_dir):
    data_args = []
    for frame_id, frame_info in tqdm(data["frame_info"].items()):
        for cam_id, frame_cam_data in frame_info["camera_image"].items():
            calib = frame_cam_data["calibration"]
            im_rel_path = frame_cam_data["path"]
            data_args.append( ( frame_info["3d_boxes"], calib[cam_id], args.data_root / im_rel_path, ensured_path(save_dir / cam_id) / f"{frame_id}.jpg", ) )
    maybe_multiprocessing(_plot_bbox_2d_of_single_frame, data_args, num_processes=args.num_workers, use_tqdm=True, tqdm_desc="plotting bbox 2d")


def plot_polyline_bev(data, save_dir):
    data_args = [ (frame_info["3d_polylines"], frame_info["ego_pose"], save_dir / f"{frame_id}.png") for frame_id, frame_info in tqdm(data["frame_info"].items()) ]
    maybe_multiprocessing(_plot_polyline_bev_of_single_frame, data_args, num_processes=args.num_workers, use_tqdm=True, tqdm_desc="plotting bbox")


def plot_polyline_2d(data, save_dir):
    calib = data["scene_info"]["calibration"]
    data_args = []
    for frame_id, frame_info in tqdm(data["frame_info"].items()):
        for cam_id, im_rel_path in frame_info["camera_image"].items():
            data_args.append( ( frame_info["3d_polylines"], calib[cam_id], args.data_root / im_rel_path, ensured_path(save_dir / cam_id) / f"{frame_id}.jpg", ) )
    maybe_multiprocessing(_plot_polyline_2d_of_single_frame, data_args, num_processes=args.num_workers, use_tqdm=True, tqdm_desc="plotting polyline 2d")


if __name__ == "__main__":
    main()
