#!/bin/bash

scene_names="
        20230820_105813
        "
fish_camera_ids="camera1 camera11 camera12 camera15 camera2 camera3 camera4 camera5 camera6 camera7 camera8"
num_cores=10

process_folder() {
    local scene_name_=$1
    for fish_camera_id in $fish_camera_ids
    do
        s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/camera/$fish_camera_id/* ./data/MV4D_12V3L/$scene_name_/camera/$fish_camera_id
    done
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/sdf/sdf_2d_-15_-15_15_15/* ./data/MV4D_12V3L/$scene_name_/sdf/sdf_2d_-15_-15_15_15
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/occ/occ_2d/occ_map_sdf_-15_-15_15_15/* ./data/MV4D_12V3L/$scene_name_/occ/occ_2d/occ_map_sdf_-15_-15_15_15
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/ground/ground_height_map_-15_-15_15_15/* ./data/MV4D_12V3L/$scene_name_/ground/ground_height_map_-15_-15_15_15
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/lidar/undistort_static_merged_lidar1/* ./data/MV4D_12V3L/$scene_name_/lidar/undistort_static_merged_lidar1
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/4d_anno_infos/* ./data/MV4D_12V3L/$scene_name_/4d_anno_infos
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/trajectory.txt ./data/MV4D_12V3L/$scene_name_/
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/calibration_center.yml ./data/MV4D_12V3L/$scene_name_/
    s5cmd --credentials-file ~/.aws/credentials --endpoint-url http://192.168.23.242:8009 --numworkers 32 --retry-count 100 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/$scene_name_/calibration_back.yml ./data/MV4D_12V3L/$scene_name_/
    python tools/dataset_converters/gene_info_4d_v2.py $scene_name_
    rm -rf ./data/MV4D_12V3L/$scene_name_/undistort_static_merged_lidar1
    # rm -rf ./data/MV4D_12V3L/$scene_name_/lidar
    
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
