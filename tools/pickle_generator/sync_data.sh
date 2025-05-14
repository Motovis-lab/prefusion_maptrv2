old_53=(
  20230820_105813 20230820_131402 20230822_110856 20230822_154430
  20230823_110018
  20230823_162939 20230824_115840 20230824_134824 20230826_102054 20230826_122208 20230829_170053
  20230830_120142 20230830_181107 20230831_101527 20230831_151057 20230901_123031 20230901_152553
  20230902_142849 20230903_123057 20231010_141702 20231027_185823 20231028_124504 20231028_134141
  20231028_150815 20231028_185150 20231031_133418 20231031_134214 20231031_135230 20231031_145557
  20231101_172858 20231102_144626 20231103_133206 20231103_140855 20231103_173838 20231104_155224
  20231105_130142 20231105_161937 20231107_123645 20231107_150715 20231107_152423 20231107_154446
  20231107_183700 20231107_212029 20231108_143610 20231108_153013 20231109_143452
  )
fix_11=(
  20231103_174738 20231101_160337 20231029_195612 20231028_134843 20231028_145730 20230822_104856
  20230829_115909 20230830_141232 20230901_121703 20231010_131855 20231105_114621 )
new_25=(20230823_113013
  20230824_172019 20230826_133639 20230828_124528 20230901_110728 20230901_120226 20230903_175455
   20231011_202057 20231028_142902 20231028_170049 20231029_123632 20231029_161907 20231103_123359
   20231104_123013 20231105_143227 20231105_195823 20231106_124102 20231107_124257 20231107_185947
   20231107_205844 20231107_220705 20231108_144706 20231108_155610 20231108_170045 20231108_204111 )
new_13=(
  20230825_110210 20230829_122655 20230831_110634 20231011_150326 20231028_125529 20231028_133437
  20231029_194228 20231031_144111 20231101_150226 20231102_151151 20231104_115532 20231107_163857
  20231108_164010 )

new_2025=(
20250315_153606 20250315_154251 20250315_154856 20250315_162419 20250315_153359
20250315_155248
20250315_155653_1742025413764_1742025533764
20250315_155653_1742025533764_1742025653764
20250315_155653_1742025653764_1742025691764
)
new_0416_16=(
            20250315_154730
            20250315_154940
            20250315_161940
            20250315_160135
            20250315_162031
            20250321_111631
            20250321_112847
            20250321_131221
            20250321_160904
            20250321_170901
            20250322_112035
            20250322_162647
            20250322_125640
            20250321_131754
            20250315_153742_1742024262664_1742024382664
            20250315_153742_1742024382664_1742024467964
            20250315_161540_1742026540764_1742026660764
            20250315_161540_1742026660764_1742026776664
#            20250315_153742
#            20250315_161540
)

new_0506=(
20231104_182238
20231031_164806
20231029_164037
20250321_111904_1742527144664_1742527264664
20250321_111904_1742527264664_1742527280664
20250321_112503_1742527504364_1742527624364
20250321_112503_1742527624364_1742527697064
20250322_131445
20250322_132133
20250322_161902_1742631542564_1742631662564
20250322_161902_1742631662564_1742631697964
20231029_174117
20250315_160423
20250322_125312_1742619193464_1742619313464
20250322_125312_1742619313464_1742619397064
20250322_164802_1742633283264_1742633403364
20250322_164802_1742633403364_1742633521864
20250322_171422
20250315_153029
20250315_154342
20250315_160629
20250322_112304
20250322_163340
)
all_file_list=("${old_53[@]}" "${new_25[@]}" "${new_13[@]}" "${fix_11[@]}"  "${new_2025[@]}" "${new_0416_16[@]}" "${new_0506[@]}")
new_ids=("${new_25[@]}" "${new_13[@]}" "${fix_11[@]}"  "${new_2025[@]}" "${new_0416_16[@]}" "${new_0506[@]}")
#new_ids=(20231108_204111)
#new_ids=(20231107_124257 20231108_170045)
#new_ids=(20231028_145730)
#for SCENE_ID in "${old_53[@]}"; do
#    echo $SCENE_ID
#    DATA_ROOT=/ssd1/MV4D_12V3L
#    SCENE_ROOT=${DATA_ROOT}/${SCENE_ID}
#    PKL_PATH1=${SCENE_ROOT}/fb_${SCENE_ID}.pkl
#    PKL_PATH2=${SCENE_ROOT}/fb_fix_${SCENE_ID}.pkl
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/lidar/undistort_static_merged_lidar1/* ${SCENE_ROOT}/lidar/undistort_static_merged_lidar1/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/camera/* ${SCENE_ROOT}/camera/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/sdf/* ${SCENE_ROOT}/sdf/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/occ/* ${SCENE_ROOT}/occ/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/4d_anno_infos/4d_anno_infos_frame/frames_labels/* ${SCENE_ROOT}/4d_anno_infos/4d_anno_infos_frame/frames_labels/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/trajectory.txt ${SCENE_ROOT}/
##    python tools/pickle_generator/generate_lidar_vcam.py --scene_root $SCENE_ROOT --pkl_save_path $PKL_PATH1
#done
#exit 0

for SCENE_ID in "${new_0506[@]}"; do
    echo $SCENE_ID
    DATA_ROOT=/ssd1/MV4D_12V3L
    SCENE_ROOT=${DATA_ROOT}/${SCENE_ID}
    PKL_PATH1=${SCENE_ROOT}/fb_${SCENE_ID}.pkl
    PKL_PATH2=${SCENE_ROOT}/fb_fix_${SCENE_ID}.pkl
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/lidar/undistort_static_merged_lidar1/* ${SCENE_ROOT}/lidar/undistort_static_merged_lidar1/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/camera/* ${SCENE_ROOT}/camera/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/sdf/* ${SCENE_ROOT}/sdf/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/occ/* ${SCENE_ROOT}/occ/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync  s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/4d_anno_infos/4d_anno_infos_frame/frames_labels/* ${SCENE_ROOT}/4d_anno_infos/4d_anno_infos_frame/frames_labels/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/trajectory.txt ${SCENE_ROOT}/
##    python tools/pickle_generator/generate_lidar_vcam.py --scene_root $SCENE_ROOT --pkl_save_path $PKL_PATH1
#    python tools/pickle_generator/gene_info_4d_fb_dag.py --scene_root $SCENE_ROOT --pkl_save_path $PKL_PATH1
#    python tools/pickle_generator/fix_pkl_from_wh.py --data-root $DATA_ROOT --scene-name $SCENE_ID --output_pickle_path $PKL_PATH2 --input_pickle_path $PKL_PATH1
#    mv ${SCENE_ROOT}/fb_train_old.pkl ${SCENE_ROOT}/fb_${SCENE_ID}.pkl
#    mv ${SCENE_ROOT}/fb_train_lidar.pkl ${SCENE_ROOT}/fb_fix_${SCENE_ID}.pkl
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/camera/* ${SCENE_ROOT}/camera/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/V* ${SCENE_ROOT}/camera/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/fb_train_old.pkl ${SCENE_ROOT}/fb_${SCENE_ID}.pkl
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/fb_train_lidar.pkl ${SCENE_ROOT}/fb_fix_${SCENE_ID}.pkl
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/lidar/undistort_static_merged_lidar1_model/* ${SCENE_ROOT}/lidar/undistort_static_merged_lidar1_model/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/trajectory.txt ${SCENE_ROOT}/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/calibration_center.yml ${SCENE_ROOT}/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/4d_anno_infos/4d_anno_infos_frame/* ${SCENE_ROOT}/4d_anno_infos/4d_anno_infos_frame/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/occ/* ${SCENE_ROOT}/occ/
    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/ground/ground_height_map_-15_-15_15_15/* ${SCENE_ROOT}/ground/ground_height_map_-15_-15_15_15/

#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/4d_anno_infos/annos.json ${SCENE_ROOT}/4d_anno_infos/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 sync s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/camera/* ${SCENE_ROOT}/camera/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/lidar/undistort_static_merged_lidar1/* ${SCENE_ROOT}/lidar/undistort_static_merged_lidar1/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 sync s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/trajectory.txt ${SCENE_ROOT}/
##    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 sync s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/calibration_center.yml ${SCENE_ROOT}/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 sync s3://mv-4d-annotation/data/MV4D_12V3L/${SCENE_ID}/4d_anno_infos/annos.json ${SCENE_ROOT}/4d_anno_infos/


#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 rm  s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/camera*
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/camera/VCAMERA_FISHEYE_BACK/ s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/VCAMERA_FISHEYE_BACK/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/camera/VCAMERA_FISHEYE_FRONT/ s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/VCAMERA_FISHEYE_FRONT/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/camera/VCAMERA_FISHEYE_LEFT/ s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/VCAMERA_FISHEYE_LEFT/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/camera/VCAMERA_FISHEYE_RIGHT/ s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/camera/VCAMERA_FISHEYE_RIGHT/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/lidar/undistort_static_merged_lidar1_model/ s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/lidar/undistort_static_merged_lidar1_model/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.243:8009 sync ${SCENE_ROOT}/fb_${SCENE_ID}.pkl s3://mv-4d-annotation/data/mvtrain/${SCENE_ID}/fb_train_old.pkl

#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp ${SCENE_ROOT}/trajectory.txt ${SCENE_ROOT}/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp ${SCENE_ROOT}/calibration_center.yml ${SCENE_ROOT}/
#    s5cmd --credentials-file ~/.aws/credentials224 --endpoint-url http://192.168.23.242:8009 cp --sp ${SCENE_ROOT}/4d_anno_infos/annos.json ${SCENE_ROOT}/4d_anno_infos/


#    python tools/pickle_generator/generate_lidar_vcam.py --scene_root $SCENE_ROOT --pkl_save_path $PKL_PATH1
#    python tools/pickle_generator/gene_info_4d_fb_dag.py --scene_root $SCENE_ROOT --pkl_save_path $PKL_PATH1
#    python tools/pickle_generator/fix_pkl_from_wh.py --data-root $DATA_ROOT --scene-name $SCENE_ID --output_pickle_path $PKL_PATH2 --input_pickle_path $PKL_PATH1
#    rsync -avP /ssd1/MV4D_12V3L/${SCENE_ID}/ wuhan@192.168.3.148:~/MV4D_12V3L/${SCENE_ID}/
#    rsync -avP /ssd1/MV4D_12V3L/${SCENE_ID}/occ/ wuhan@192.168.3.148:~/MV4D_12V3L/${SCENE_ID}/occ/
#    rsync -avP /ssd1/MV4D_12V3L/${SCENE_ID}/ground/ wuhan@192.168.3.148:~/MV4D_12V3L/${SCENE_ID}/ground/
#    rsync -avP /ssd1/MV4D_12V3L/${SCENE_ID}/camera/V* wuhan@192.168.3.148:~/MV4D_12V3L/${SCENE_ID}/camera/
#    rsync -avP /ssd1/MV4D_12V3L/${SCENE_ID}/lidar/undistort_static_merged_lidar1_model/ wuhan@192.168.3.148:~/MV4D_12V3L/${SCENE_ID}/lidar/undistort_static_merged_lidar1_model/

#--data-root /ssd1/MV4D_12V3L --scene-name 20230823_113013  --input_pickle_path /ssd1/MV4D_12V3L/20230823_113013/fb_20230823_113013.pkl --output_pickle_path /ssd1/MV4D_12V3L/20230823_113013/fb_fix_20230823_113013.pkl
##    python3 process_file.py --filel_name "$file"

done
