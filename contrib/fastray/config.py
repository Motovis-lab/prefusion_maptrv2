experiment_name = "fastray_planar_demo"

_base_ = "../../../configs/default_runtime.py"

custom_imports = dict(
    imports=["prefusion", "contrib.fastray"], 
    allow_failed_imports=False
)



train_dataset = dict(
    type='GroupBatchDataset',
    name="DemoParkingDataset",
    data_root='/home/alpha/Projects/MV4D-PARKING/20231028_150815',
    info_path='/home/alpha/Projects/MV4D-PARKING/info_pkls/mv_4d_infos_20231028_150815.pkl',
    model_feeder=dict(
        type="FastRayModelFeeder"
    )
)




train_dataloader = dict(
    num_workers=1,
    persistent_workers=True,
    collate_fn=dict(type="collate_dict"),
    dataset=dict(
        type="GroupBatchDataset",
        name="ParkingDataset",
        data_root="/data/datasets/mv4d",
        info_path="/data/datasets/mv4d/mv4d_infos_dbg_246_noalign.pkl",
        model_feeder=dict(
            type="StreamPETRModelFeeder",
            visible_range=point_cloud_range,
        ),
        transformables=dict(
            camera_images=dict(
                type='camera_images', 
                tensor_smith=dict(
                    type='CameraImageTensor',
                    means=[123.675, 116.280, 103.530],
                    stds=[58.395, 57.120, 57.375],
                )
            ),
            bbox_3d_0=dict(type='bbox_3d', dictionary={'classes': det_classes}),
            bbox_3d_1=dict(type='bbox_3d', dictionary={'classes': det_classes})
        )
        transforms=[
            # dict(type="RandomMirrorSpace", prob=0.5, scope="group"),
            dict(
                type="RandomImageISP",
                prob=0.0001,
            ),
        ],
        phase="train",
        batch_size=batch_size,
        possible_group_sizes=[1],
        possible_frame_intervals=[1],
    ),
)

val_dataloader = train_dataloader
test_dataloader = train_dataloader