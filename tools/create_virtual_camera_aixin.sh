#!/bin/bash

scene_root=$1; shift;
num_workers=$1; shift;
num_frames_to_process=${1:-60}; shift;


################################
# Output Perspective Cameras
################################
python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/mv_calibration_back.yaml \
    --img-suffix jpg \
    --real-camera-id camera_XFV_FRONT \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera_XFV_FRONT.png \
    --real-camera-image-dir ${scene_root}/cam_sy_x8b_front \
    --rotation-euler-angles -90 0 0 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 2228 896 \
    --virtual-camera-image-crop-params 420 0 1344 896 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/VCAMERA_PERSPECTIVE_FRONT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT.yml \
    --num-frames-to-process ${num_frames_to_process} \
    --num-workers ${num_workers}
    # --virtual-camera-size 1592 640 \
    # --virtual-camera-image-crop-params 300 0 960 640 \


python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/mv_calibration_back.yaml \
    --img-suffix jpg \
    --real-camera-id camera_XPilot_FISHEYE_LEFT \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera_XPilot_FISHEYE_LEFT.png \
    --real-camera-image-dir ${scene_root}/cam_sy_x3j_avm_left \
    --rotation-euler-angles -90 7 75 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT_LEFT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 1344 1148 \
    --virtual-camera-image-crop-params 0 0 1344 896 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT_LEFT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/VCAMERA_PERSPECTIVE_FRONT_LEFT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_LEFT.yml \
    --num-frames-to-process ${num_frames_to_process} \
    --num-workers ${num_workers}
    # --virtual-camera-size 960 820 \
    # --virtual-camera-image-crop-params 0 0 960 640 \

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/mv_calibration_back.yaml \
    --img-suffix jpg \
    --real-camera-id camera_XPilot_FISHEYE_RIGHT \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera_XPilot_FISHEYE_RIGHT.png \
    --real-camera-image-dir ${scene_root}/cam_sy_x3j_avm_right \
    --rotation-euler-angles -90 -7 -75 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT_RIGHT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 1344 1148 \
    --virtual-camera-image-crop-params 0 0 1344 896 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT_RIGHT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/VCAMERA_PERSPECTIVE_FRONT_RIGHT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_RIGHT.yml \
    --num-frames-to-process ${num_frames_to_process} \
    --num-workers ${num_workers}
    # --virtual-camera-size 960 820 \
    # --virtual-camera-image-crop-params 0 0 960 640 \

python tools/merge_calibration_files.py \
    --input-calib-files \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_LEFT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_RIGHT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT.yml \
    --output-calib-file \
        ${scene_root}/vcamera_calibration/vcalib_merged_back.yml
