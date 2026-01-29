import torch
from functools import partial
import numpy as np
from scipy.optimize import curve_fit

def proj_func(x, params):
    p0, p1, p2, p3 = params
    return x + p0 * x**3 + p1 * x**5 + p2 * x**7 + p3 * x**9


def poly_odd6(x, k0, k1, k2, k3, k4, k5):
    return x + k0 * x**3 + k1 * x**5 + k2 * x**7 + k3 * x**9 + k4 * x**11 + k5 * x**13


def get_unproj_func(p0, p1, p2, p3, fov=200):
    theta = np.linspace(-0.5 * fov * np.pi / 180,  0.5 * fov * np.pi / 180, 2000)
    theta_d = proj_func(theta, (p0, p1, p2, p3))
    params, pcov = curve_fit(poly_odd6, theta_d, theta)
    error = np.sqrt(np.diag(pcov)).mean()
    assert error < 1e-2, "poly parameter curve fitting failed: {:f}.".format(error)
    k0, k1, k2, k3, k4, k5 = params
    return partial(poly_odd6, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def intrinsics_matrix(fx, fy, cx, cy):
    K = np.eye(3, dtype=float)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K 