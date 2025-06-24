"""
Noisy Instance Generator for Denoising Training (DN) in DETR-like models.

This module implements the logic for generating noisy instances during training
to improve model convergence through denoising objectives.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class NoisyInstanceGenerator(nn.Module):
    """
    Generates noisy instances for denoising training in DETR-like models.
    
    This class handles:
    1. Creating multiple noisy copies of ground truth bboxes
    2. Adding Gaussian noise to bbox centers
    3. Creating attention masks to prevent inappropriate query interactions
    4. Handling temporal modeling with memory components
    
    Args:
        num_classes (int): Number of object classes
        num_query (int): Number of object queries
        num_propagated (int): Number of propagated queries from memory
        memory_len (int): Length of memory buffer
        scalar (int): Number of noisy copies to create for each GT bbox
        bbox_noise_scale (float): Scale factor for bbox center noise
        bbox_noise_trans (float): Translation offset for noise
        split (float): Threshold for determining noisy vs clean labels
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    
    def __init__(
        self,
        num_classes: int,
        num_query: int = 100,
        num_propagated: int = 256,
        memory_len: int = 1024,
        scalar: int = 5,
        bbox_noise_scale: float = 0.4,
        bbox_noise_trans: float = 0.0,
        split: float = 0.5,
        pc_range: List[float] = [-65, -65, -8.0, 65, 65, 8.0]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_query = num_query
        self.num_propagated = num_propagated
        self.memory_len = memory_len
        self.scalar = scalar
        self.bbox_noise_scale = bbox_noise_scale
        self.bbox_noise_trans = bbox_noise_trans
        self.split = split
        
        # Register pc_range as buffer so it's part of the module state
        self.register_buffer('pc_range', torch.tensor(pc_range))
    
    def generate_noisy_instances(
        self,
        batch_size: int,
        reference_points: torch.Tensor,
        gt_bboxes_3d: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        training: bool = True,
        with_dn: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Generate noisy instances for denoising training.
        
        Args:
            batch_size (int): Batch size
            reference_points (torch.Tensor): Reference points tensor [num_query, 3]
            gt_bboxes_3d (List[torch.Tensor]): List of GT 3D bboxes for each batch
            gt_labels (List[torch.Tensor]): List of GT labels for each batch
            training (bool): Whether in training mode
            with_dn (bool): Whether to use denoising training
            
        Returns:
            Tuple containing:
            - padded_reference_points (torch.Tensor): Reference points with DN padding
            - attn_mask (Optional[torch.Tensor]): Attention mask for DN queries
            - mask_dict (Optional[Dict]): Dictionary containing DN metadata
        """
        if not (training and with_dn):
            return self._create_non_dn_output(batch_size, reference_points)
        
        # Step 1: Create validity masks for GT instances
        gt_validity_masks = self._create_gt_validity_masks(gt_labels)
        gt_counts_per_batch = [t.size(0) for t in gt_bboxes_3d]
        
        # Step 2: Concatenate all GT data across batches
        all_gt_labels = torch.cat(gt_labels)
        all_gt_bboxes = torch.cat(gt_bboxes_3d)
        batch_indices = self._create_batch_indices(gt_bboxes_3d)
        
        # Step 3: Generate noisy copies
        noisy_data = self._generate_noisy_copies(
            all_gt_labels, all_gt_bboxes, batch_indices, reference_points.device
        )
        
        # Step 4: Create padded reference points
        max_gt_per_batch = max(gt_counts_per_batch) if gt_counts_per_batch else 0
        dn_pad_size = max_gt_per_batch * self.scalar
        padded_reference_points = self._create_padded_reference_points(
            reference_points, batch_size, dn_pad_size, 
            noisy_data, gt_counts_per_batch, max_gt_per_batch
        )
        
        # Step 5: Create attention masks
        attn_mask = self._create_attention_masks(
            dn_pad_size, max_gt_per_batch, reference_points.device
        )
        
        # Step 6: Create metadata dictionary
        mask_dict = self._create_mask_dict(
            noisy_data, batch_indices, gt_counts_per_batch, 
            max_gt_per_batch, gt_validity_masks, dn_pad_size
        )
        
        return padded_reference_points, attn_mask, mask_dict
    
    def _create_non_dn_output(
        self, 
        batch_size: int, 
        reference_points: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """Create output for non-denoising case."""
        padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        return padded_reference_points, None, None
    
    def _create_gt_validity_masks(self, gt_labels: List[torch.Tensor]) -> List[torch.Tensor]:
        """Create validity masks for ground truth instances."""
        return [(torch.ones_like(t)).cuda() for t in gt_labels]
    
    def _create_batch_indices(self, gt_bboxes_3d: List[torch.Tensor]) -> torch.Tensor:
        """Create batch indices for each GT instance."""
        return torch.cat([
            torch.full((t.size(0),), i) 
            for i, t in enumerate(gt_bboxes_3d)
        ])
    
    def _generate_noisy_copies(
        self,
        all_gt_labels: torch.Tensor,
        all_gt_bboxes: torch.Tensor,
        batch_indices: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Generate noisy copies of GT instances."""
        # Get all valid GT indices
        valid_gt_mask = torch.ones_like(all_gt_labels)
        valid_indices = torch.nonzero(valid_gt_mask).view(-1)
        
        # Repeat indices and data for multiple noisy copies
        repeated_indices = valid_indices.repeat(self.scalar, 1).view(-1)
        repeated_labels = all_gt_labels.repeat(self.scalar, 1).view(-1).long().to(device)
        repeated_batch_idx = batch_indices.repeat(self.scalar, 1).view(-1)
        repeated_bboxes = all_gt_bboxes.repeat(self.scalar, 1).to(device)
        
        # Extract center and scale for noise generation
        noisy_centers = repeated_bboxes[:, :3].clone()
        bbox_scales = repeated_bboxes[:, 3:6].clone()
        
        # Add noise to centers if noise scale > 0
        if self.bbox_noise_scale > 0:
            noisy_centers, repeated_labels = self._add_noise_to_centers(
                noisy_centers, bbox_scales, repeated_labels
            )
        
        return {
            'valid_indices': repeated_indices,
            'labels': repeated_labels,
            'batch_indices': repeated_batch_idx,
            'bboxes': repeated_bboxes,
            'centers': noisy_centers
        }
    
    def _add_noise_to_centers(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to bbox centers."""
        # Calculate noise variance based on bbox size
        noise_variance = scales / 2 + self.bbox_noise_trans
        
        # Generate random noise in [-1, 1]
        random_noise = torch.rand_like(centers) * 2 - 1.0
        
        # Apply noise to centers
        centers += torch.mul(random_noise, noise_variance) * self.bbox_noise_scale
        
        # Normalize centers to [0, 1] range
        centers[..., 0:3] = (centers[..., 0:3] - self.pc_range[0:3]) / (
            self.pc_range[3:6] - self.pc_range[0:3]
        )
        centers = centers.clamp(min=0.0, max=1.0)
        
        # Mark heavily noised instances as background class
        noise_magnitude_mask = torch.norm(random_noise, 2, 1) > self.split
        labels[noise_magnitude_mask] = self.num_classes
        
        return centers, labels
    
    def _create_padded_reference_points(
        self,
        reference_points: torch.Tensor,
        batch_size: int,
        dn_pad_size: int,
        noisy_data: Dict[str, torch.Tensor],
        gt_counts_per_batch: List[int],
        max_gt_per_batch: int
    ) -> torch.Tensor:
        """Create padded reference points with DN queries."""
        # Create zero padding for DN queries
        padding_bbox = torch.zeros(dn_pad_size, 3).to(reference_points.device)
        padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0)
        padded_reference_points = padded_reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Fill in noisy centers if we have GT data
        if gt_counts_per_batch:
            position_mapping = self._create_position_mapping(gt_counts_per_batch, max_gt_per_batch)
            batch_indices = noisy_data['batch_indices']
            
            if len(batch_indices):
                padded_reference_points[
                    (batch_indices.long(), position_mapping)
                ] = noisy_data['centers'].to(reference_points.device)
        
        return padded_reference_points
    
    def _create_position_mapping(
        self, 
        gt_counts_per_batch: List[int], 
        max_gt_per_batch: int
    ) -> torch.Tensor:
        """Create mapping from flattened indices to padded positions."""
        # Create base mapping for one scalar copy
        base_mapping = torch.cat([
            torch.tensor(range(num)) for num in gt_counts_per_batch
        ])
        
        # Extend mapping for all scalar copies
        position_mapping = torch.cat([
            base_mapping + max_gt_per_batch * i 
            for i in range(self.scalar)
        ]).long()
        
        return position_mapping
    
    def _create_attention_masks(
        self,
        dn_pad_size: int,
        max_gt_per_batch: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create attention masks for DN queries."""
        # Basic mask for DN + regular queries
        basic_size = dn_pad_size + self.num_query
        attn_mask = torch.ones(basic_size, basic_size).to(device) < 0
        
        # Regular queries cannot see DN queries
        attn_mask[dn_pad_size:, :dn_pad_size] = True
        
        # DN queries from different scalar copies cannot see each other
        for i in range(self.scalar):
            start_idx = max_gt_per_batch * i
            end_idx = max_gt_per_batch * (i + 1)
            
            if i == 0:
                # First copy cannot see later copies
                attn_mask[start_idx:end_idx, end_idx:dn_pad_size] = True
            elif i == self.scalar - 1:
                # Last copy cannot see earlier copies
                attn_mask[start_idx:end_idx, :start_idx] = True
            else:
                # Middle copies cannot see any other copies
                attn_mask[start_idx:end_idx, end_idx:dn_pad_size] = True
                attn_mask[start_idx:end_idx, :start_idx] = True
        
        # Extend mask for temporal modeling
        query_size = dn_pad_size + self.num_query + self.num_propagated
        target_size = dn_pad_size + self.num_query + self.memory_len
        temporal_attn_mask = torch.ones(query_size, target_size).to(device) < 0
        
        # Copy basic mask to temporal mask
        temporal_attn_mask[:basic_size, :basic_size] = attn_mask
        temporal_attn_mask[dn_pad_size:, :dn_pad_size] = True
        
        return temporal_attn_mask
    
    def _create_mask_dict(
        self,
        noisy_data: Dict[str, torch.Tensor],
        batch_indices: torch.Tensor,
        gt_counts_per_batch: List[int],
        max_gt_per_batch: int,
        gt_validity_masks: List[torch.Tensor],
        dn_pad_size: int
    ) -> Dict[str, Any]:
        """Create metadata dictionary for DN training."""
        position_mapping = self._create_position_mapping(gt_counts_per_batch, max_gt_per_batch)
        
        return {
            'known_indice': torch.as_tensor(noisy_data['valid_indices']).long(),
            'batch_idx': torch.as_tensor(batch_indices).long(),
            'map_known_indice': torch.as_tensor(position_mapping).long(),
            'known_lbs_bboxes': (noisy_data['labels'], noisy_data['bboxes']),
            'know_idx': gt_validity_masks,
            'pad_size': dn_pad_size
        }
