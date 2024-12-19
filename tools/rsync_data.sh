# 先传数据
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
# 20230826_102054
# 20230826_122208
# 20230828_134528
# 20230830_120142
# 20230830_181107
# 20230831_101527
# 20230831_151057
# 20230901_123031
# 20230901_152553
# 20230902_142849
# 20231010_141702
# 20231027_185823
# 20231028_124504
# 20231028_134141
# 20231028_134843
# 20231028_145730
# 20231028_150815
# 20231028_185150
# 20231029_195612
# 20231031_133418
# 20231031_134214
# 20231031_135230
# 20231031_144111
# 20231101_150226
# 20231101_160337
# 20231101_172858
# 20231102_144626
# 20231103_133206
# 20231103_140855
# 20231103_173838
# 20231103_174738
# 20231104_115532
# 20231104_155224
# 20231105_161937
# 20231107_123645
# 20231107_152423
# 20231107_154446
# 20231107_183700
# 20231107_212029
# 20231108_143610
# 20231108_153013
# 再传pkl，调换scene_names就OK
scene_names="
        mv_4d_infos_20231028_124504.pkl
        mv_4d_infos_20231029_195612.pkl
        mv_4d_infos_20230826_122208.pkl
        mv_4d_infos_20231028_145730.pkl
        mv_4d_infos_20231101_172858.pkl
        mv_4d_infos_20230828_134528.pkl
        mv_4d_infos_20231103_133206.pkl
        mv_4d_infos_20230831_101527.pkl
        mv_4d_infos_20231028_134843.pkl
        mv_4d_infos_20231107_154446.pkl
        mv_4d_infos_20231028_134141.pkl
        mv_4d_infos_20231101_150226.pkl
        mv_4d_infos_20230820_131402.pkl
        mv_4d_infos_20231104_115532.pkl
        mv_4d_infos_20231108_143610.pkl
        mv_4d_infos_20230822_154430.pkl
        mv_4d_infos_20230830_120142.pkl
        mv_4d_infos_20230820_105813.pkl
        mv_4d_infos_20230822_104856.pkl
        mv_4d_infos_20231107_123645.pkl
        mv_4d_infos_20230830_181107.pkl
        mv_4d_infos_20231103_173838.pkl
        mv_4d_infos_20230901_123031.pkl
        mv_4d_infos_20231105_161937.pkl
        mv_4d_infos_20231028_150815.pkl
        mv_4d_infos_20230823_110018.pkl
        mv_4d_infos_20230902_142849.pkl
        mv_4d_infos_20231107_152423.pkl
        mv_4d_infos_20231031_134214.pkl
        mv_4d_infos_20231031_144111.pkl
        mv_4d_infos_20230824_153239.pkl
        mv_4d_infos_20231103_174738.pkl
        mv_4d_infos_20231102_144626.pkl
        mv_4d_infos_20230901_152553.pkl
        mv_4d_infos_20231031_133418.pkl
        mv_4d_infos_20231104_155224.pkl
        mv_4d_infos_20231010_141702.pkl
        mv_4d_infos_20230824_115840.pkl
        mv_4d_infos_20230823_162939.pkl
        mv_4d_infos_20231028_185150.pkl
        mv_4d_infos_20231103_140855.pkl
        mv_4d_infos_20231107_212029.pkl
        mv_4d_infos_20231108_153013.pkl
        mv_4d_infos_20230826_102054.pkl
        mv_4d_infos_20231027_185823.pkl
        mv_4d_infos_20230831_151057.pkl
        mv_4d_infos_20231107_183700.pkl
        mv_4d_infos_20230824_134824.pkl
        mv_4d_infos_20230822_110856.pkl
        mv_4d_infos_20231031_135230.pkl
        mv_4d_infos_20231101_160337.pkl
        "
# 先把所有的数据都软连接到这个文件夹下，全部数据可能一个位置放不下
SOURCE_DIR="/mnt/ssd1/wuhan/prefusion/data/MV4D_12V3L"
num_cores=5

process_folder(){
    local scene_name=$1
    # 传数据
    # tar --dereference -zcf "$scene_name.tar.gz" -C "$SOURCE_DIR" "$scene_name"
    # rsync -avzP "$scene_name.tar.gz" wuhan@192.168.3.148:/share/home/wuhan/MV4D_12V3L/
    # rm "$scene_name.tar.gz"
    # 传pkl
    rsync -avzP "$SOURCE_DIR/$scene_name" wuhan@192.168.3.148:/share/home/wuhan/MV4D_12V3L/
    echo "$SOURCE_DIR/$scene_name"
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