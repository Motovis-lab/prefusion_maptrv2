#!/bin/bash

scene_root=$1; shift;
num_workers=$1; shift;

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera1 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera1.png \
    --real-camera-image-dir ${scene_root}/camera/camera1 \
    --rotation-euler-angles -120 0 180 \
    --virtual-camera-id VCAMERA_FISHEYE_BACK \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_FISHEYE_BACK \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_FISHEYE_BACK.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_BACK.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera5 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera5.png \
    --real-camera-image-dir ${scene_root}/camera/camera5 \
    --rotation-euler-angles -135 0 90 \
    --virtual-camera-id VCAMERA_FISHEYE_LEFT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_FISHEYE_LEFT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_FISHEYE_LEFT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_LEFT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera8 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera8.png \
    --real-camera-image-dir ${scene_root}/camera/camera8 \
    --rotation-euler-angles -120 0 0 \
    --virtual-camera-id VCAMERA_FISHEYE_FRONT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_FISHEYE_FRONT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_FISHEYE_FRONT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_FRONT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera11 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera11.png \
    --real-camera-image-dir ${scene_root}/camera/camera11 \
    --rotation-euler-angles -135 0 -90 \
    --virtual-camera-id VCAMERA_FISHEYE_RIGHT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_FISHEYE_RIGHT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_FISHEYE_RIGHT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_RIGHT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera2 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera2.png \
    --real-camera-image-dir ${scene_root}/camera/camera2 \
    --rotation-euler-angles -90 0 40 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT_LEFT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT_LEFT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_FRONT_LEFT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_LEFT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera4 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera4.png \
    --real-camera-image-dir ${scene_root}/camera/camera4 \
    --rotation-euler-angles -90 0 -40 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT_RIGHT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT_RIGHT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_FRONT_RIGHT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_RIGHT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera3 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera3.png \
    --real-camera-image-dir ${scene_root}/camera/camera3 \
    --rotation-euler-angles -90 0 135 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_BACK_LEFT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_BACK_LEFT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_BACK_LEFT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK_LEFT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera7 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera7.png \
    --real-camera-image-dir ${scene_root}/camera/camera7 \
    --rotation-euler-angles -90 0 -135 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_BACK_RIGHT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_BACK_RIGHT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_BACK_RIGHT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK_RIGHT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera6 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera6.png \
    --real-camera-image-dir ${scene_root}/camera/camera6 \
    --rotation-euler-angles -90 0 0 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 640 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_FRONT.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT.yml \
    --num-workers ${num_workers}

python tools/create_virtual_camera.py \
    --motovis-calibration ${scene_root}/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera12 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask ${scene_root}/self_mask/camera/camera12.png \
    --real-camera-image-dir ${scene_root}/camera/camera12 \
    --rotation-euler-angles -90 0 180 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_BACK \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 512 320 \
    --virtual-camera-image-save-dir ${scene_root}/vcamera/VCAMERA_PERSPECTIVE_BACK \
    --virtual-camera-mask-save-path ${scene_root}/self_mask/camera/VCAMERA_PERSPECTIVE_BACK.png \
    --virtual-camera-calibration-save-path ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK.yml \
    --num-workers ${num_workers}

python tools/merge_calibration_files.py \
    --input-calib-files \
        ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_BACK.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_FRONT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_LEFT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_FISHEYE_RIGHT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK_LEFT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK_RIGHT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_BACK.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_LEFT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT_RIGHT.yml \
        ${scene_root}/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT.yml \
    --output-calib-file \
        ${scene_root}/vcalib_center.yml
