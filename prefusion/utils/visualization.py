from typing import Tuple, Union, List

import numpy as np
from copious.cv.camera_model import FisheyeCameraModel
from copious.cv.geometry import rt2mat
from shapely.geometry import MultiPoint, box


def K3x3(cx, cy, fx, fy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class PointNotOnImageError(Exception):
    pass

def pinhole_project(points_homo: np.ndarray, extrinsic: Tuple[np.ndarray, np.ndarray], intrinsic: np.ndarray, im_size: Tuple[int, int], conservative=True) -> np.ndarray:
    """project 3d points onto image

    Parameters
    ----------
    points_homo : np.ndarray
        of shape (N, 4)
    extrinsic : Tuple[np.ndarray, np.ndarray]
        of shape (3, 3) and (3, 1), denoting rotation matrix and translation vector
    intrinsic : np.ndarray
        of shape (4,), representing pp and focal
    im_size : Tuple[int, int]
        (width, height)
    conservative : bool, optional
        _description_, by default True

    Returns
    -------
    np.ndarray
        uv on the image, of shape (N, 2)
    """
    cam_extr, cam_intr = rt2mat(*extrinsic, as_homo=True), K3x3(*intrinsic[:4])
    T_cam_ego = np.linalg.inv(cam_extr)
    cam_coords = (T_cam_ego @ points_homo.T).T[:, :3]
    cam_coords = check_camera_coords_visibility_on_image(cam_coords, conservative)
    normalized_cam_coords = cam_coords[:, :2] / cam_coords[:, 2:3]
    im_coords = (cam_intr[:2, :2] @ normalized_cam_coords.T).T + cam_intr[:2, 2]
    im_coords = check_im_coords_visibility_on_image(im_coords, im_size, conservative)
    return im_coords


def fisheye_project(points_homo: np.ndarray, extrinsic: Tuple[np.ndarray, np.ndarray], intrinsic: np.ndarray, im_size: Tuple[int, int], conservative=True) -> np.ndarray:
    pp, focal, inv_poly = intrinsic[:2], intrinsic[2:4], intrinsic[4:]
    cam_extr = rt2mat(*extrinsic, as_homo=True)
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
            raise PointNotOnImageError
    else:
        if (cam_coords[:, 2] < 0).all():
            raise PointNotOnImageError
        cam_coords = cam_coords[cam_coords[:, 2] >= 0]
    return cam_coords


def check_im_coords_visibility_on_image(im_coords, im_size, conservative=True):
    if conservative:
        if not im_pts_within_image(im_coords, im_size).all():
            raise PointNotOnImageError
    else:
        if not im_pts_within_image(im_coords, im_size).any():
            raise PointNotOnImageError
        im_coords = im_coords[im_pts_within_image(im_coords, im_size)]
    return im_coords


def im_pts_within_image(pts, im_size):
    w, h = im_size
    return (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)


def corner_pts_to_img_bbox(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
