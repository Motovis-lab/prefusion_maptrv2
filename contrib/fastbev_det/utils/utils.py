import numpy as np
from pyquaternion import Quaternion

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


def get_cam_corners(corners, translation, rotation, cam_intrinsics, mv4d=False):
    cam_corners = corners.copy()
    cam_corners -= np.array(translation)
    cam_corners = cam_corners @ Quaternion(rotation).inverse.rotation_matrix.T
    cam_corners = cam_corners @ np.array(cam_intrinsics).T
    valid = cam_corners[:, -1] > 0
    cam_corners /= cam_corners[:, 2:3]
    cam_corners[~valid] = 0
    return cam_corners

def intrinsics_matrix(intrinsic):
    cx, cy, fx, fy = intrinsic
    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

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

def get_bev_lines_cylinder(corners):
    return [[[corners[i, 0], corners[(i + 1) % 4, 0]],
             [corners[i, 1], corners[(i + 1) % 4, 1]]] for i in range(100)]