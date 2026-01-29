from contrib.petr import StreamPETRHead
from mmengine.registry import MODELS
import torch
from contrib.petr.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from contrib.petr.misc import normalize_bbox, bias_init_with_prob, MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from copy import deepcopy   
from torch import nn


__all__ = ['RaySPETRHead']

@MODELS.register_module()
class RaySPETRHead(StreamPETRHead):
    def __init__(self, *args, **kwargs):
        super(RaySPETRHead, self).__init__(*args, **kwargs)

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))
        temp_pos_ = temp_pos.clone()
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_pos_[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose
    
    def forward(self, img_feats, memory_center, img_metas, gt_bboxes_3d, gt_labels, topk_indexes=None, centerness=None, **data):
        """_summary_

        Parameters
        ----------
        img_feats : _type_
            (B, N, C, H, W).
        memory_center : _type_
            _description_
        img_metas : _type_
            _description_
        gt_bboxes_3d : _type_
            _description_
        gt_labels : _type_
            _description_
        topk_indexes : _type_, optional
            _description_, by default None
            
        Returns
        -------
        _type_
            _description_
        """
        # zero init the memory bank
        with torch.amp.autocast("cuda", enabled=False):
            self.pre_update_memory(data)

        x = img_feats
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)
        query_ind = centerness.topk(self.num_query, dim=1).indices if centerness is not None else None
        init_query_pos = topk_gather(memory, query_ind) if query_ind is not None else None

        pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)
        memory = self.memory_embed(memory)

        # spatial_alignment in focal petr
        memory = self.spatial_alignment(memory, cone) # MLN
        pos_embed = self.featurized_pe(pos_embed, memory)

        # reference_points = self.reference_points.weight
        dn_reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, init_query_pos.device, gt_bboxes_3d, gt_labels)
        query_pos = torch.concat([self.query_embedding(pos2posemb3d(dn_reference_points)), init_query_pos], dim=1) if init_query_pos is not None else self.query_embedding(pos2posemb3d(dn_reference_points))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, query_pos)

        # transformer here is a little different from PETR
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            # reference = inverse_sigmoid(reference_points.clone())
            # assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])  # reg offsets of reference points  or  direct to the bbox center

            # tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])  # expand to the original range

        # update the memory bank
        with torch.amp.autocast("cuda", enabled=False):
            self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,
            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
            }

        return outs
    
    def prepare_for_dn(self, batch_size, device, gt_bboxes_3d, gt_labels):
        if self.training and self.with_dn:
            known = [(torch.ones_like(t)).cuda() for t in gt_labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in gt_bboxes_3d]

            gt_labels = torch.cat([t for t in gt_labels])
            boxes = torch.cat([t for t in gt_bboxes_3d])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(gt_bboxes_3d)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = gt_labels.repeat(self.scalar, 1).view(-1).long().to(device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(device)
            padded_reference_points = padding_bbox.repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }

        else:
            assert False, "Only support training with dn now."

        return padded_reference_points, attn_mask, mask_dict