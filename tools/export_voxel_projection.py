import sys

import torch.onnx.operators
sys.path.append("/home/wuhan/prefusion/")


from contrib.fastbev_det.models.necks.view_transform import VoxelProjection_fish, VoxelProjection_pv, VoxelProjection_front
from contrib.fastbev_det.models.backbones import FastRay_DP
from prefusion.registry import MODELS
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
import torch.nn as nn 

cfg = Config.fromfile("contrib/fastbev_det/configs/mv_4d_fastbev_3dbox_deploy.py")

input_fish = torch.ones((4, 3, 80, 128)).float().cuda()
input_pv = torch.ones(5, 3, 96, 128).float().cuda()
input_front = torch.ones(1, 3, 96, 192).float().cuda()

input_depth_fish = torch.ones(4, 142, 80, 128).float().cuda() - 0.5
input_depth_pv = torch.ones(5, 142, 96, 128).float().cuda() - 0.5
input_depth_front = torch.ones(1, 142, 96, 192).float().cuda() - 0.5
img_front = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_0.jpg").transpose(2,0,1)
img_left = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_1.jpg").transpose(2,0,1)
img_back = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_2.jpg").transpose(2,0,1)
img_right = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_3.jpg").transpose(2,0,1)

input = torch.from_numpy(np.stack([img_front, img_left, img_back, img_right], axis=0)).cuda() / 255.0

model = MODELS.build(cfg.model.backbone_conf).cuda()
model.forward = model.pure_forward
# out = model.eval()(input_fish, input_pv, input_front)
# out = out.view(18,240,120)
# tmp = torch.stack([out[0], out[6], out[12]], dim=0).cpu().numpy()
# plt.imsave("work_dirs/vt_debug/fish_out.jpg", tmp.transpose(1,2,0))

save_root = "./work_dirs/deploy/voxel_projection_conv_64_permute_int.onnx"
save_root = "./work_dirs/deploy/voxel_projection_conv_64_permute_int_dd.onnx"

torch.onnx.export(model, (input_fish, input_pv, input_front, input_depth_fish, input_depth_pv, input_depth_front), save_root, opset_version=11,  # input must be tuple type
        input_names = ['input_fish', 'input_pv', 'input_front', 'input_depth_fish', 'input_depth_pv', 'input_depth_front'],
        output_names = ['output'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

from onnxsim import simplify
import onnx
# 加载导出的 ONNX 模型
onnx_model = onnx.load(save_root)

# 简化模型
simplified_model, check = simplify(onnx_model)

# 保存简化后的模型
onnx.save_model(simplified_model, save_root,)
print("ONNX file saved in {}".format(save_root))

print('finished')

