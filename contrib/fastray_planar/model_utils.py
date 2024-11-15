
import torch
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'draw_out_feats',
    'draw_aligned_voxel_feats',
    'get_bbox_3d',
    'get_parkingslot_3d'
]


def draw_out_feats(
        batched_input_dict, 
        camera_tensors_dict,
        pred_bbox_3d,
        pred_polyline_3d=None,
        pred_parkingslot_3d=None,
    ):

    fig, _ = plt.subplots(3, 10)
    fig.suptitle(batched_input_dict['index_infos'][0].scene_frame_id)
    for i, cam_id in enumerate(camera_tensors_dict):
        img = camera_tensors_dict[cam_id].detach().cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1] * 255 + 128
        img = img.astype(np.uint8)
        plt.subplot(3, 10, i+1)
        plt.title(cam_id.replace('VCAMERA_', '').lower())
        plt.imshow(img)

    gt_seg = batched_input_dict['annotations']['bbox_3d']['seg'][0][0].detach().cpu()
    pred_seg = pred_bbox_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
    plt.subplot(3, 10, 11)
    plt.imshow(gt_seg)
    plt.title('bbox_3d gt_seg')
    plt.subplot(3, 10, 12)
    plt.imshow(pred_seg)
    plt.title("bbox_3d pred_seg")
    
    gt_cen = batched_input_dict['annotations']['bbox_3d']['cen'][0][0].detach().cpu()
    pred_cen = pred_bbox_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
    pred_cen *= (pred_seg > 0.5)
    plt.subplot(3, 10, 13)
    plt.imshow(gt_cen)
    plt.title("bbox_3d gt_cen")
    plt.subplot(3, 10, 14)
    plt.imshow(pred_cen)
    plt.title("bbox_3d pred_cen")
    
    gt_reg = batched_input_dict['annotations']['bbox_3d']['reg'][0][0].detach().cpu()
    pred_reg = pred_bbox_3d['reg'][0][0].to(torch.float32).detach().cpu()
    pred_reg *= (pred_seg > 0.5)
    plt.subplot(3, 10, 15)
    plt.imshow(gt_reg)
    plt.title("bbox_3d gt_reg")
    plt.subplot(3, 10, 16)
    plt.imshow(pred_reg)
    plt.title("bbox_3d pred_reg")
    
    if pred_polyline_3d is not None:
        gt_seg = batched_input_dict['annotations']['polyline_3d']['seg'][0][0].detach().cpu()
        pred_seg = pred_polyline_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 17)
        plt.imshow(gt_seg)
        plt.title('polyline_3d gt_seg')
        plt.subplot(3, 10, 18)
        plt.imshow(pred_seg)
        plt.title("polyline_3d pred_seg")
        
        gt_reg = batched_input_dict['annotations']['polyline_3d']['reg'][0][0].detach().cpu()
        pred_reg = pred_polyline_3d['reg'][0][0].to(torch.float32).detach().cpu()
        pred_reg *= (pred_seg > 0.5)
        plt.subplot(3, 10, 19)
        plt.imshow(gt_reg)
        plt.title("polyline_3d gt_reg")
        plt.subplot(3, 10, 20)
        plt.imshow(pred_reg)
        plt.title("polyline_3d pred_reg")
    
    if pred_parkingslot_3d is not None:
        gt_seg = batched_input_dict['annotations']['parkingslot_3d']['seg'][0][1].detach().cpu()
        pred_mask = pred_parkingslot_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_seg = pred_parkingslot_3d['seg'][0][1].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 21)
        plt.imshow(gt_seg)
        plt.title('parkingslot_3d gt_seg')
        plt.subplot(3, 10, 22)
        plt.imshow(pred_seg)
        plt.title("parkingslot_3d pred_seg")
        
        gt_cen = batched_input_dict['annotations']['parkingslot_3d']['cen'][0][0].detach().cpu()
        pred_cen = pred_parkingslot_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_cen *= (pred_mask > 0.5)
        plt.subplot(3, 10, 23)
        plt.imshow(gt_cen)
        plt.title("parkingslot_3d gt_cen")
        plt.subplot(3, 10, 24)
        plt.imshow(pred_cen)
        plt.title("parkingslot_3d pred_cen")
        
        gt_reg = batched_input_dict['annotations']['parkingslot_3d']['reg'][0][2].detach().cpu()
        pred_reg = pred_parkingslot_3d['reg'][0][2].to(torch.float32).detach().cpu()
        pred_reg *= (pred_mask > 0.5)
        plt.subplot(3, 10, 25)
        plt.imshow(gt_reg)
        plt.title("parkingslot_3d gt_reg")
        plt.subplot(3, 10, 26)
        plt.imshow(pred_reg)
        plt.title("parkingslot_3d pred_reg")
    

    voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
    plt.subplot(3, 10, 27)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    gt_boxes_3d = batched_input_dict['transformables'][0]['bbox_3d']
    
    for element in gt_boxes_3d.elements:
        center = element['translation'][:, 0]
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('gt: ', corner_points[:, :2])
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'g')
    
    # gt_boxes_3d = batched_input_dict['annotations']['bbox_3d']
    plt.subplot(3, 10, 28)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    pred_bbox_3d_0 = {
        'cen': pred_bbox_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_bbox_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_bbox_3d['reg'][0].cpu().float()
    }
    pred_boxes_3d = get_bbox_3d(pred_bbox_3d_0)
    
    for element in pred_boxes_3d:
        # if element['confs'][0] < 0.7:
        #     continue
        if element['area_score'] < 0.5:
            continue
        center = element['translation']
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('pred: ', corner_points[:, :2])
        plt.text(center[1], center[0], '{:.2f}'.format(element['area_score']), color='r',
                 ha='center', va='center')
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'r')
    
    plt.subplot(3, 10, 29)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])

    gt_slots_3d = batched_input_dict['transformables'][0]['parkingslot_3d']
    
    for element in gt_slots_3d.elements:
        points = element['points']
        plt.plot(points[[1, 2, 3, 0], 1], points[[1, 2, 3, 0], 0], 'g')
    
    
    plt.subplot(3, 10, 30)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    pred_parkingslot_3d_0 = {
        'cen': pred_parkingslot_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_parkingslot_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_parkingslot_3d['reg'][0].cpu().float()
    }
    
    pred_slots_3d = get_parkingslot_3d(pred_parkingslot_3d_0)
    # print(pred_slots_3d)
    for slot in pred_slots_3d:
        # plt.text(points[0, 1], points[0, 0], '{:.2f}'.format(element['confs'][0]), color='r')
        plt.plot(slot[[1, 2, 3, 0], 1], slot[[1, 2, 3, 0], 0], 'r')
    
    plt.show()


def draw_aligned_voxel_feats(aligned_voxel_feats):

    n_frames = len(aligned_voxel_feats)
    plt.subplots(1, n_frames)
    for i, voxel_feats_frame in enumerate(aligned_voxel_feats):
        plt.subplot(1, n_frames, i+1)
        plt.imshow(voxel_feats_frame[0, 0].detach().cpu().to(torch.float32).numpy() > 0)
        plt.title(f'frame t({0 - i})')
    plt.show()


def get_bbox_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarBbox3D
    pbox = PlanarBbox3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.3,
        reverse_nms_ratio=1.0
    )
    return pbox.reverse(tensor_dict)
    

def get_parkingslot_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarParkingSlot3D
    pslot = PlanarParkingSlot3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.5
    )
    return pslot.reverse(tensor_dict)


