# import debugpy
# debugpy.connect(('localhost', 5678))

import argparse
from pathlib import Path
from typing import List

from loguru import logger
import mmcv
import numpy as np
from scipy.spatial.transform import Rotation
from copious.io.fs import parent_ensured_path, ensured_path, read_yaml, write_yaml
from copious.io.parallelism import maybe_multiprocessing
from virtual_camera import FisheyeCamera, PerspectiveCamera, render_image

CAMERA_MODEL_CLS_MAPPING = {"PerspectiveCamera": PerspectiveCamera, "FisheyeCamera": FisheyeCamera}


def _render_virtual_image(
    real_im_path,
    real_cam_model,
    real_camera_calib,
    virtual_camera_type,
    virtual_im_save_path,
    virtual_mask_save_path,
    rotation_euler_angle,
    W,
    H,
):
    src_image = mmcv.imread(real_im_path, channel_order="bgr")
    R = Rotation.from_euler("xyz", angles=rotation_euler_angle, degrees=True).as_matrix()
    t = [0, 0, 0]
    v_cam_rmatrix = R
    v_cam_t = np.array(real_camera_calib["extrinsic"][:3]).reshape(3)
    if virtual_camera_type == "PerspectiveCamera":
        fov = 90
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
        vcamera = PerspectiveCamera((W, H), (R, t), intrinsic)
    elif virtual_camera_type == "FisheyeCamera":
        fov = 180
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
        vcamera = FisheyeCamera((W, H), (R, t), intrinsic, fov=fov)
    dst_image, dst_mask = render_image(src_image, real_cam_model, vcamera)
    mmcv.imwrite(dst_image, virtual_im_save_path)
    mmcv.imwrite(dst_mask, virtual_mask_save_path)

    return v_cam_rmatrix, v_cam_t, intrinsic, src_image, real_cam_model, vcamera, fov


def get_real_image_paths(real_camera_image_dir: Path, img_suffix: str) -> List[Path]:
    return sorted(real_camera_image_dir.glob(f"*.{img_suffix.strip('.')}"))


def create_camera_model(cam_model_type: str, cam_calib_info: dict, ego_mask_path: Path = None):
    cam_model_cls = CAMERA_MODEL_CLS_MAPPING[cam_model_type]
    cam_model = cam_model_cls.init_from_motovis_cfg(cam_calib_info)
    if ego_mask_path is not None:
        _mask = mmcv.imread(ego_mask_path, channel_order="gray")
        if _mask.ndim == 3:
            _mask = _mask[..., 0]
        if _mask.max() == 255:
            _mask = _mask / 255
        cam_model.ego_mask = _mask
    return cam_model


def render_virtual_iamge(data_args):
    return _render_virtual_image(*data_args)
    # (
    #     real_im_path,
    #     real_cam_model,
    #     real_cam_calib,
    #     virtual_camera_type,
    #     virtual_im_save_path,
    #     virtual_camera_mask_save_path,
    #     rotation_euler_angles,
    #     *virtual_camera_size,
    # ) = data_args
    
    # return _render_virtual_image(
    #     real_im_path,
    #     real_cam_model,
    #     real_cam_calib,
    #     virtual_camera_type,
    #     virtual_im_save_path,
    #     virtual_camera_mask_save_path,
    #     rotation_euler_angles,
    #     *virtual_camera_size,
    # )


def main(args):
    calib = read_yaml(args.motovis_calibration)
    real_cam_model = create_camera_model(args.real_camera_type, calib["rig"][args.real_camera_id], ego_mask_path=args.real_camera_ego_mask)
    real_image_paths = get_real_image_paths(args.real_camera_image_dir, args.img_suffix)
    data_args = [
        (
            p,
            real_cam_model,
            calib["rig"][args.real_camera_id],
            args.virtual_camera_type,
            args.virtual_camera_image_save_dir / p.name,
            args.virtual_camera_mask_save_path,
            args.rotation_euler_angles,
            *args.virtual_camera_size,
        )
        for p in real_image_paths
    ]
    res = maybe_multiprocessing(
        render_virtual_iamge, data_args, args.num_workers, use_tqdm=True, tqdm_desc="Rendering virtual images..."
    )

    v_camera_rmatrix, v_camera_t, v_camera_intrinsic, d_src_image, d_real_cam_model, d_vcamera, vcamera_fov = res[-1]
    vcam_calib = {
        "extrinsic": v_camera_t.tolist() + Rotation.from_matrix(v_camera_rmatrix).as_quat().tolist(),
        "pp": list(v_camera_intrinsic[:2]),
        "focal": list(v_camera_intrinsic[2:4]),
        "image_size": list(d_vcamera.resolution),
        "fov_fit": vcamera_fov,
        "sensor_model": type(d_vcamera).__name__,
        "sensor_id": args.virtual_camera_id,
    }
    if args.virtual_camera_type == "FisheyeCamera":
        vcam_calib["inv_poly"] = list(v_camera_intrinsic[4:])
    write_yaml(vcam_calib, args.virtual_camera_calibration_save_path)


if __name__ == "__main__":
    # "--rotation-euler-angles", "-90", "0", "45",   // camera8  -> VCAMERA_PERSPECTIVE_FRONT_LEFT
    # "--rotation-euler-angles", "-90", "0", "-45",  // camera8  -> VCAMERA_PERSPECTIVE_FRONT_RIGHT
    # "--rotation-euler-angles", "-90", "0", "135",  // camera1  -> VCAMERA_PERSPECTIVE_BACK_LEFT
    # "--rotation-euler-angles", "-90", "0", "-135", // camera1  -> VCAMERA_PERSPECTIVE_BACK_RIGHT
    # "--rotation-euler-angles", "-90", "0", "90",   // camera5  -> VCAMERA_PERSPECTIVE_LEFT_FORWARD
    # "--rotation-euler-angles", "-90", "0", "-90",  // camera11 -> VCAMERA_PERSPECTIVE_RIGHT_FORWARD
    parser = argparse.ArgumentParser()
    parser.add_argument("--motovis-calibration", type=Path, required=True)
    parser.add_argument("--img-suffix", default="jpg", required=True)
    parser.add_argument("--real-camera-id", type=str, required=True)
    parser.add_argument("--real-camera-type", choices=["PerspectiveCamera", "FisheyeCamera"], required=True)
    parser.add_argument("--real-camera-ego-mask", type=Path)
    parser.add_argument("--real-camera-image-dir", type=Path, required=True)
    parser.add_argument("--rotation-euler-angles", nargs=3, type=float, required=True, help="基于初始状态(XYZ对应自车前左上)进行旋转")
    parser.add_argument("--virtual-camera-id", type=str, required=True)
    parser.add_argument("--virtual-camera-type", choices=["PerspectiveCamera", "FisheyeCamera"], required=True)
    parser.add_argument("--virtual-camera-size", nargs=2, type=int, required=True, help="--virtual-camera-size <width> <height>")
    parser.add_argument("--virtual-camera-image-save-dir", type=ensured_path, required=True)
    parser.add_argument("--virtual-camera-mask-save-path", type=parent_ensured_path, required=True)
    parser.add_argument("--virtual-camera-calibration-save-path", type=parent_ensured_path, required=True)
    parser.add_argument("--num-workers", default=0, type=int)
    main(parser.parse_args())
