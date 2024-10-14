import sys
sys.path.append("/home/wuhan/prefusion/")


from contrib.fastbev_det.models.necks.view_transform import VoxelProjection_fish
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt


input = torch.ones((4,3,80,128)).float().cuda()
img_front = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_0.jpg").transpose(2,0,1)
img_left = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_1.jpg").transpose(2,0,1)
img_back = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_2.jpg").transpose(2,0,1)
img_right = mmcv.imread("work_dirs/vt_debug/img_fish_feats_0_3.jpg").transpose(2,0,1)

input = torch.from_numpy(np.stack([img_front, img_left, img_back, img_right], axis=0)).cuda() / 255.0

model = VoxelProjection_fish()
out = model.eval()(input)
# out = out.view(18,240,120)
# tmp = torch.stack([out[0], out[6], out[12]], dim=0).cpu().numpy()
# plt.imsave("work_dirs/vt_debug/fish_out.jpg", tmp.transpose(1,2,0))

torch.onnx.export(model, input, "./work_dirs/deploy/voxel_projection.onnx", opset_version=11,
        input_names = ['input'],
        output_names = ['output'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)