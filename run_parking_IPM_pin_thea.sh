#!/bin/bash
CUDA_VISIBLE_DEVICES=5 bash tools/dist_train_5090.sh contrib/fastray_planar/configs/fastray_planar_single_frame_park_apa_scaled_relu6_5090_parking_IPM_pin_test.py 1 #--amp  #--cfg-options compile=True 
