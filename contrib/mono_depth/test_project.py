import mmengine
import numpy as np
import virtual_camera
import matplotlib.pyplot as plt 
import pdb


frame_info = mmengine.load("/home/wuhan/prefusion/data/146_data/mv_4d_infos_20231108_153013.pkl")
frame_0 = frame_info["20231108_153013"]['frame_info']['1699428613664']
frame_1 = frame_info["20231108_153013"]['frame_info']['1699428614664']
frame_2 = frame_info["20231108_153013"]['frame_info']['1699428615664']

frame_0_img = np.load(frame_0['camera_image_depth']['VCAMERA_FISHEYE_FRONT'])['depth'].astype(np.float32)
frame_1_img = np.load(frame_1['camera_image_depth']['VCAMERA_FISHEYE_FRONT'])['depth'].astype(np.float32)
frame_2_img = np.load(frame_2['camera_image_depth']['VCAMERA_FISHEYE_FRONT'])['depth'].astype(np.float32)

plt.imshow(frame_0_img); plt.show()
plt.imshow(frame_1_img); plt.show()
plt.imshow(frame_2_img); plt.show()