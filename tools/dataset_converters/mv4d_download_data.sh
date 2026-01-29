#!/bin/bash

# 20230820_131402
# 20230820_105813
# 20230822_104856
# 20230822_110856
# 20230822_154430
# 20230823_110018
# 20230823_162939
# 20230824_115840
# 20230824_134824
# 20230824_153239

scene_names="
        20230826_102054
        20230826_122208
        20230828_134528
        20230830_120142
        20230830_181107
        20230831_101527
        20230831_151057
        20230901_123031
        20230901_152553
        20230902_142849
        20231010_141702
        20231027_185823
        20231028_124504
        20231028_134141
        20231028_134843
        20231028_145730
        20231028_150815
        20231028_185150
        20231029_195612
        20231031_133418
        20231031_134214
        20231031_135230
        20231031_144111
        20231101_150226
        20231101_160337
        20231101_172858
        20231102_144626
        20231103_133206
        20231103_140855
        20231103_173838
        20231103_174738
        20231104_115532
        20231104_155224
        20231105_161937
        20231107_123645
        20231107_152423
        20231107_154446
        20231107_183700
        20231107_212029
        20231108_143610
        20231108_153013
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
    # python tools/dataset_converters/gene_info_4d_v2.py $scene_name_
    # rm -rf ./data/MV4D_12V3L/$scene_name_/undistort_static_merged_lidar1
    
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
