#!/bin/bash

scene_names="
        20230829_134351
        20230829_122655
        20230828_141035
        20230828_134528
        20230828_124528
        20230826_145040
        20230826_133639
        20230826_124414
        20230826_122208
        20230826_104049
        20230826_102054
        20230826_100858
        20230826_095802
        20230825_140415
        20230825_135721
        20230825_131119
        20230825_110210
        20230824_172019
        20230824_170838
        20230824_153239
        20230824_134824
        20230824_125254
        20230824_115840
        20230823_162939
        20230823_160913
        20230823_150601
        20230823_121926
        20230823_113013
        20230823_110018
        20230823_101949
        20230822_155910
        20230822_154430
        20230822_132203
        20230822_110856
        20230822_104856
        20230822_102201
        20230821_155824
        20230821_144803
        20230821_124733
        20230821_104222
        20230820_144650
        20230820_135628
        20230820_131402
        20230820_110545
        20230820_105813
        20230820_100405
        "
fish_camera_ids="camera1 camera5 camera8 camera11"
perspective_camera_ids="camera6"
num_cores=$(nproc)

process_folder() {
    local scene_name_=$1
    for fish_camera_id in $fish_camera_ids
    do
        s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/multimodel_data_baidu/$scene_name_/camera/$fish_camera_id/* ./data/pretrain_data/$scene_name_/camera/$fish_camera_id
        s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/multimodel_data_baidu/$scene_name_/seg/fisheye_semantic_segmentation/$fish_camera_id/* ./data/pretrain_data/$scene_name_/seg/fisheye_semantic_segmentation/$fish_camera_id
    done
    for perspective_camera_id in $perspective_camera_ids
    do
        s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/multimodel_data_baidu/$scene_name_/camera/$perspective_camera_id/* ./data/pretrain_data/$scene_name_/camera/$perspective_camera_id
        s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/multimodel_data_baidu/$scene_name_/seg/perspective_semantic_segmentation/$perspective_camera_id/* ./data/pretrain_data/$scene_name_/seg/perspective_semantic_segmentation/$perspective_camera_id
    done
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/multimodel_anno_baidu/$scene_name_/undistort_static_merged_lidar1/* ./data/pretrain_data/$scene_name_/undistort_static_merged_lidar1
    python tools/dataset_converters/pretrain_gene_info_4d.py $scene_name_
    rm -rf ./data/pretrain_data/$scene_name_/undistort_static_merged_lidar1
    rm -rf ./data/pretrain_data/$scene_name_/lidar
    echo "processed $scene_name"
}


for scene_name in $scene_names
do
    echo $scene_name
    process_folder "$scene_name" &
    
    ((job_count++))
    
    if [ $job_count -eq $num_cores ]; then
        wait -n
        ((job_count--))
    fi
done
