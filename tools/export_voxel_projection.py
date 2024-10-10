from contrib.fastbev_det.models.necks.view_transform import VoxelProjection
import torch



input = torch.ones((4,336,80,128)).float().cuda()
model = VoxelProjection()
out = model.eval()(input)

torch.onnx.export(model, input, "./work_dirs/deploy/voxel_projection.onnx", opset_version=11,
        input_names = ['input'],
        output_names = ['output'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)