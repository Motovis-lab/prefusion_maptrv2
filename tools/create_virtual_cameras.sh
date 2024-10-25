#!/bin/bash

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera1 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera1.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera1 \
    --rotation-euler-angles -120 0 180 \
    --virtual-camera-id VCAMERA_FISHEYE_BACK \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_FISHEYE_BACK \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_FISHEYE_BACK.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_FISHEYE_BACK.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera5 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera5.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera5 \
    --rotation-euler-angles -135 0 90 \
    --virtual-camera-id VCAMERA_FISHEYE_LEFT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_FISHEYE_LEFT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_FISHEYE_LEFT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_FISHEYE_LEFT.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera8 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera8.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera8 \
    --rotation-euler-angles -120 0 0 \
    --virtual-camera-id VCAMERA_FISHEYE_FRONT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_FISHEYE_FRONT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_FISHEYE_FRONT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_FISHEYE_FRONT.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera11 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera11.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera11 \
    --rotation-euler-angles -135 0 -90 \
    --virtual-camera-id VCAMERA_FISHEYE_RIGHT \
    --virtual-camera-type FisheyeCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_FISHEYE_RIGHT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_FISHEYE_RIGHT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_FISHEYE_RIGHT.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera2 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera2.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera2 \
    --rotation-euler-angles -90 0 40 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_LEFT_FRONT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_PERSPECTIVE_LEFT_FRONT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_PERSPECTIVE_LEFT_FRONT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_PERSPECTIVE_LEFT_FRONT.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera4 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera4.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera4 \
    --rotation-euler-angles -90 0 -40 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_RIGHT_FRONT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_PERSPECTIVE_RIGHT_FRONT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_PERSPECTIVE_RIGHT_FRONT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_PERSPECTIVE_RIGHT_FRONT.yml \
    --num-workers 10

python tools/create_virtual_camera.py \
    --motovis-calibration /data/datasets/mv4d/20231101_160337/calibration_center.yml \
    --img-suffix jpg \
    --real-camera-id camera6 \
    --real-camera-type FisheyeCamera \
    --real-camera-ego-mask /data/datasets/mv4d/20231101_160337/self_mask/camera/camera6.png \
    --real-camera-image-dir /data/datasets/mv4d/20231101_160337/camera/camera6 \
    --rotation-euler-angles -90 0 0 \
    --virtual-camera-id VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-type PerspectiveCamera \
    --virtual-camera-size 1024 640 \
    --virtual-camera-image-save-dir /data/datasets/mv4d/20231101_160337/vcamera/VCAMERA_PERSPECTIVE_FRONT \
    --virtual-camera-mask-save-path /data/datasets/mv4d/20231101_160337/self_mask/camera/VCAMERA_PERSPECTIVE_FRONT.png \
    --virtual-camera-calibration-save-path /data/datasets/mv4d/20231101_160337/vcamera_calibration/VCAMERA_PERSPECTIVE_FRONT.yml \
    --num-workers 10
