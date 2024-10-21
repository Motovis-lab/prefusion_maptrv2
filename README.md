
[![pipeline status](https://gitlab.com/auto-ai-ragtag/prefusion/badges/main/pipeline.svg)](https://gitlab.com/auto-ai-ragtag/prefusion/-/commits/main)

[![coverage report](https://gitlab.com/auto-ai-ragtag/prefusion/badges/main/coverage.svg)](https://gitlab.com/auto-ai-ragtag/prefusion/-/commits/main)

# 前融合PreFusion


## 训练框架设计
基于mmegine来设计一个直观清晰的前融合框架，易于理解、使用、维护
  - 数据转换：可将其他开源数据转换为motovis-4d-data格式
  - 数据读取：将motovis-4d-data按需求转换成模型读取所需要的格式info.pkl
  - 数据增强：模块化的数据形态以及增强方法
  - 模型设计：兼容BEV、时空融合、Transformer等技术的模型设计，支持多帧、时序特征输入；支持量化训练
  - 任务设计：支持BEV检测、BEV矢量、3D检测、3D语义OCC、甚至端到端
  - 指标设计：各个任务的METRIC
  - 可视化训练器
  - 参数配置
  - 推理和部署：易于使用的数据推理以及ONNX部署输出

### 组件清单
- Config for Runner
  - train_dataloader
    - dataset
    - sampler
    - collate_fn
    - ...
  - val_dataloader
  - test_dataloader
  - train_cfg
    - type: GroupBatchTrainLoop
  - val_cfg
  - test_cfg
  - val_evaluator
  - model
  - data_preprocessor
  - optim_wrapper
    - AmpOptimizerWrapper
  - param_scheduler
  - default_hooks
  - launcher
  - env_cfg
  - log_level
  - load_from
  - resume
- Dataset及相关组件
  - GroupBatchDataset
  - DistributedGroupSampler
  - collate_fn 
  - Transforms
  - Transformables
- Runner
  - GroupBatchTrainLoop
  - GroupValLoop
  - GroupTestLoop
- Model
  - MatrivVT
  - WidthFormer4D
  - Backbones
- Metrics
  - val_evaluator
  - test_evaluator
- Hooks

## 数据结构设计
具体参考@袁施薇 的数据说明4d数据表示设计以及4d数据json和sdk使用说明
- <scene-name>/ 场景名
  - sensors/ 传感器类型
    - cameras/front/<timestamp>.jpg
    - camera_masks/front.png
    - lidars/LIDAR_TOP/<timestamp>.pcd
  - calibration.json
  - annotations
    - segs/front/<timestamp>.png
    - depths/front/<timestamp>.png
    - occs/ 同时存储occ和sdf
      - 2d/<timestamp>.png
      - 3d/<timestamp>.pkl
    - trajectory_ego.json
    - moving_object_tracks.json
    - objects_4d.json
    - visibility.json
    - map_4d.json
    - can_bus/

- scene_id 命名参考：
    - `mv_scene_0001`
    - `nu_scene_0001`
    - `av_scene_0001`
    - `wm_scene_0001`
    - `ax_scene_0001`

## 数据读取设计
- 开源数据到MOTOVIS-4D数据的转换，分功能实现检测、矢量图以及其他
- MOTOVIS-4D数据的读取与处理：4D数据到读取数据PKL的转换[TODO]

### 坐标系说明：
- ego: 前左上
- box: 前左上
- lidar: 前左上 

### PKL格式说明:
格式为字典:
{
    <scene_id>: scene_dict,
    <scene_id>: scene_dict,
    ...
}

对于每个`scene_dict`, 包含以下字段
- scene_info: 
  - calibration:
    - VCAMERA_PERSPECTIVE_FRONT:
      - extrinsic: (R, t)   # in forward-left-up
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_PERSPECTIVE_FRONT_LEFT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_PERSPECTIVE_FRONT_RIGHT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_PERSPECTIVE_BACK:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_PERSPECTIVE_BACK_LEFT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_PERSPECTIVE_BACK_RIGHT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy)
    - VCAMERA_FISHEYE_FRONT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    - VCAMERA_FISHEYE_LEFT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    - VCAMERA_FISHEYE_RIGHT:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    - VCAMERA_FISHEYE_BACK:
      - extrinsic: (R, t)  
      - intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    - <other cameras>
    - lidar1:
      - extrinsic: (R, t)  
    - lidar2:
      - extrinsic: (R, t)  
    - lidar3:
      - extrinsic: (R, t)  
  - camera_mask:  # ego mask and view mask for each camera
    - VCAMERA_PERSPECTIVE_FRONT: <path>
  - moving_objects_track_id_trajectory:   # available frames, in world system
    - <track_id>:  
      - class: vehicle.passenger_car
      - timestamps: [t_0, t_1, t_2, ..., t_n]  # lidar timestamp
      - poses: [(R, t), (R, t), (R, t), ..., (R, t)]  # length equals to timestamp list
      - attr:
        - vehicle.is_trunk_open:
          - [false, false, false, ..., true]  # length equals to timestamp list
        - vehicle.is_door_open:
          - [false, false, false, ..., true]  # length equals to timestamp list
    - <other tracks>
- meta_info
  - description: <tags/text>  主要是针对场景或者单帧工况的描述标签，用于给出均衡的样本，或者针对标签的测试指标等。需要定义一些标签，然后可以安排人工挑选，训练模型预标注
  - space_range:
    - map: [36, -12, -12, 12，10， -10]  # front back left right range from center
    - det:  [36, -12, -12, 12, 10,  -10]  # front back left right range from center
    - occ: [36, -12, -12, 12, 10, -10]  # front back left right range from center
  - time_range: 2 # in seconds
  - time_unit: 1e-3 # in seconds， float64
  - ori_camera: 
    - VCAMERA_FISHEYE_RIGHT: camera11
    - ...
- frame_info:
  - <timestamp_lidar>:  # key frame dict,  lidar LIDAR_TOP的timestamp
    - camera_image:
      - VCAMERA_PERSPECTIVE_FRONT: <scene_id>/sensors/cameras/VCAMERA_PERSPECTIVE_FRONT/<timestamp_camera_specific>.jpg
      - <other cameras>
    - 3d_boxes: [<3d_box>, <3d_box>, ...]  # in ego system, current frame, box with r,t and track_id
      - <3d_box>
        - class: vehicle.passenger_car   # 参考数据标注设计字典
        - attr:   # 参考数据标注设计字典
          - vehicle.is_trunk_open:
            - false
          - vehicle.is_door_open:
            - false
        - size: (l, w, h)  # l, w, h corresponds to x-y-z, forward-left-up
        - rotation:  R  # np.array
        - translation: [x, y, z]  # center point position, in ego system
        - track_id: <track_id>
        - [TODO] frame_visibility: 1
        - velocity: [vx, vy, vz]
    - 3d_polylines: [<polyline>, <polyline>, ...]  # in ego system, current frame
      - <polyline>
        - class: road_marker.lane_line   # 参考数据标注设计字典
        - attr:  # 参考数据标注设计字典, 有多少个属性，就安排多少个字典
          - road_marker.lane_line.type:
            - regular
          - road_marker.lane_line.style:
            - dashed
          - common.color.single_color:
            - yellow
        - points: [[x, y, z], [x, y, z], [x, y, z], ...]  # in np.array
    - timestamp_window:   # for indexing previous frames
      - VCAMERA_PERSPECTIVE_FRONT:
        - [<filename0>, <filename1>, ...]  # 已包含timestamp
      - <other cameras>
      - lidar1:
        - [<filename0>, <filename1>, ...] # 已包含timestamp
      - <other lidars>
    - lidar_points:   # current frame
      - LIDAR_TOP: <scene_id>/sensors/lidars/LIDAR_TOP/<timestamp>.pcd
      - LIDAR_LEFT: <scene_id>/sensors/lidars/LIDAR_LEFT/<timestamp>.pcd
      - LIDAR_RIGHT: <scene_id>/sensors/lidars/LIDAR_RIGHT/<timestamp>.pcd
    - ego_pose:  # current frame, in world system, trajectory
      - rotation:  R  # np.array
      - translation: [x, y, z]  # ego origin point
    - camera_image_seg:
      - VCAMERA_PERSPECTIVE_FRONT: <scene_id>/annotations/segs/VCAMERA_PERSPECTIVE_FRONT/<timestamp_camera_specific>.png
      - <other cameras>
    - camera_image_depth:
      - VCAMERA_PERSPECTIVE_FRONT: <scene_id>/annotations/depths/VCAMERA_PERSPECTIVE_FRONT/<timestamp_camera_specific>.png
    - occ_sdf:  # in ego system, bev
      - occ_bev: <path>
      - sdf_bev: <path>
      - height_bev: <path>
      - occ_sdf_3d:<scene_id>/annotations/occ_sdfs/<timestamp>.pkl
  - <other timestamps>:

### indices
可以准备单独的训练indices，用于手动调整数据比例
```
train.txt
<scene_id>/<timestamp_lidar>
<scene_id>/<timestamp_lidar>
<scene_id>/<timestamp_lidar>
```

## 数据增强设计
### 方法
- 图像本身的增强， ISP
- 内外参增强
  - 图、标注不变，抖动内外参数
  - 内外参数不变，根据抖动的内外参数改变图像和标注
- 自车的时空增强 RST F
  - 镜像世界：空间左右翻转
  - 镜像世界：时光回流
  - 空间旋转，整体轻微旋转 RST
  - (进阶)时间增强：对于静态目标任务，t时刻，可以输入t-dt的图像，假设知道自车轨迹，可以计算t-dt时刻传感器相对与t时刻自车坐标系的pose，输入t-dt时刻图像以及pose_(t-t_dt)，等同传感器位置变动增强
  - 放缩世界：3D空间中放大和缩小， 时间加快和减缓
- 模态省略
- (离线增强) UnifiedObjectSample，数据在线增强
- BEV特征增强
- 数据采样的增强：CBGS: Class Balanced Grouping Sampler

### 清单
- Transformables
  - camera_images: CameraImageSet
  - camera_segs: CameraSegMaskSet
  - camera_depths: CameraDepthSet
  - ego_poses: EgoPoseSet
  - lidar_points: LidarPoints
  - bbox3d: Bbox3D
  - bboxbev: BboxBev
  - cylinder3d: Cylinder3D(Bbox3D)
  - oriented_cylinder3d: OrientedCylinder3D(Bbox3D)
  - square3d: Square3D(Bbox3D)
  - polyline3d: Polyline3D
  - polygon3d: Polygon3D(Polyline3D)
  - parkingslot3d: ParkingSlot3D(Polyline3D)
  - seg_bev: SegBev
  - occ_sdf_bev: OccSdfBev
  - occ_sdf_3d: OccSdf3D
- Transforms
  - RandomImageOmit
  - RandomImageISP
  - RenderIntrinsic
  - RenderExtrinsic
  - RandomSetIntrinsicParam
  - RandomSetExtrinsicParam
  - RandomRenderExtrinsic
  - RandomMirrorSpace
  - RandomMirrorTime
  - RandomScaleSpace
  - RandomScaleTime

## 模型设计
算法改进备忘
- 基于MATRIXVT的改进
  - MATRIXVT立体感知改造灵感
  - 关于OCC3D的改造灵感这个优先级不是最高的
- ADA-TRACK
- TDA4前融合设计
  - BEV范围

## 任务设计
TODO

## 环境安装

### pip

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install openmim
mim install -r mim-requirements.txt
```

### Docker

Build from Dockerfile

```bash
docker build -t prefusion:lastest .
```

Pull from Dockerhub

```bash
docker pull brianlan/prefusion:v2
```

Run with docker

```bash
docker run --gpus 0 --privileged --shm-size=32g --rm -it --name prefusion -v /home:/home -v /data:/data brianlan/prefusion:v2 /bin/bash
```

## TODO List

- [x] 利用 Gitlab 提供的 CI/CD功能来帮我们跑测试
- [ ] 利用 Gitlab 提供的 docs 功能来帮我们部署文档
