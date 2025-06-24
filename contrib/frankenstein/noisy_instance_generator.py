"""
Noisy Instance Generator for Denoising Training (DN) in DETR-like models.

This module implements the logic for generating noisy instances during training
to improve model convergence through denoising objectives.

The core idea behind denoising training is to create a more stable and robust
learning signal for DETR-like models by giving the model practice at "denoising"
corrupted ground truth instances. This addresses the slow convergence problem
commonly seen in DETR models by providing additional supervised training signals
during early training stages.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class NoisyInstanceGenerator(nn.Module):
    """
    Generates noisy instances for denoising training in DETR-like models.
    
    WHY this approach is needed:
    - DETR models suffer from slow convergence due to sparse supervision signals
    - Traditional object detection has dense supervision (each anchor/pixel gets a label)
    - DETR only supervises a small number of queries that match ground truth
    - Denoising training provides additional supervision by training the model to
      reconstruct clean ground truth from noisy versions
    
    This class handles:
    1. Creating multiple noisy copies of ground truth bboxes (increases training signal density)
    2. Adding Gaussian noise to bbox centers (forces model to learn denoising)
    3. Creating attention masks to prevent inappropriate query interactions (maintains training stability)
    4. Handling temporal modeling with memory components (for video sequence models)
    
    Args:
        num_classes (int): Number of object classes
        num_query (int): Number of object queries  
        num_propagated (int): Number of propagated queries from memory
        memory_len (int): Length of memory buffer
        num_dn_groups (int): Number of noisy copies to create for each GT bbox (typically 5)
        bbox_noise_scale (float): Scale factor for bbox center noise (controls corruption level)
        bbox_noise_trans (float): Translation offset for noise (adds diversity to noise patterns)
        split (float): Threshold for determining noisy vs clean labels (prevents over-corruption)
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    
    def __init__(
        self,
        num_classes: int,
        num_query: int = 100,
        num_propagated: int = 256,
        memory_len: int = 1024,
        num_dn_groups: int = 5,
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
        self.num_dn_groups = num_dn_groups
        self.bbox_noise_scale = bbox_noise_scale
        self.bbox_noise_trans = bbox_noise_trans
        self.split = split
        
        # WHY register as buffer: pc_range needs to be moved to the same device as the model
        # but shouldn't be updated during training (it's a constant defining the coordinate space)
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
        
        WHY this approach works:
        - Creates multiple noisy versions of each GT instance to provide richer supervision
        - Forces the model to learn robust feature representations that can handle noise
        - Provides additional training targets beyond the sparse GT matching in DETR
        
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
        # WHY: We need to track which GT instances are valid vs padding in batched processing
        gt_validity_masks = self._create_gt_validity_masks(gt_labels)
        gt_counts_in_the_batch = [t.size(0) for t in gt_bboxes_3d]
        
        # Step 2: Concatenate all GT data across samples in a batch
        # WHY: Processing all GT instances together simplifies noise generation and indexing
        all_gt_labels = torch.cat(gt_labels)
        all_gt_bboxes = torch.cat(gt_bboxes_3d)
        batch_indices = self._create_batch_indices(gt_bboxes_3d)
        
        # Step 3: Generate noisy copies
        noisy_data = self._generate_noisy_copies(
            all_gt_labels, all_gt_bboxes, batch_indices, reference_points.device
        )
        
        # Step 4: Create padded reference points
        # WHY: DN queries are prepended to regular queries, requiring careful indexing
        # The padding ensures all DN queries fit before the regular model queries
        max_gt_per_batch = max(gt_counts_in_the_batch) if gt_counts_in_the_batch else 0
        dn_pad_size = max_gt_per_batch * self.num_dn_groups
        padded_reference_points = self._create_padded_reference_points(
            reference_points, batch_size, dn_pad_size, 
            noisy_data, gt_counts_in_the_batch, max_gt_per_batch
        )
        
        # Step 5: Create attention masks
        # WHY: Different DN groups must not see each other to prevent information leakage
        # This maintains the denoising challenge while allowing proper gradient flow
        attn_mask = self._create_attention_masks(
            dn_pad_size, max_gt_per_batch, reference_points.device
        )
        
        # Step 6: Create metadata dictionary
        mask_dict = self._create_mask_dict(
            noisy_data, batch_indices, gt_counts_in_the_batch, 
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
        """Create validity masks for ground truth instances.
        
        WHY: In batched processing, some samples may have fewer GT instances than others.
        These masks help track which instances are real vs padding during loss computation.
        """
        return [(torch.ones_like(t)).cuda() for t in gt_labels]
    
    def _create_batch_indices(self, gt_bboxes_3d: List[torch.Tensor]) -> torch.Tensor:
        """Create batch indices for each GT instance.
        
        WHY: After concatenating GT data across batches, we need to remember which
        batch each GT instance came from for proper loss computation and indexing.
        """
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
        """Generate noisy copies of GT instances.
        
        WHY: Creating multiple noisy versions of each GT provides more training signal.
        Instead of only having sparse GT matches, the model gets practice reconstructing
        clean targets from multiple noise levels, accelerating convergence.
        """
        # Get all valid GT indices
        # WHY: Using ones_like creates a mask that's True for all valid GT instances
        # This handles the case where some GTs might be invalid/padding in the future
        valid_gt_mask = torch.ones_like(all_gt_labels)
        valid_indices = torch.nonzero(valid_gt_mask).view(-1)
        
        # Repeat indices and data for multiple noisy copies
        # WHY: Each GT instance is replicated num_dn_groups times to create multiple
        # noise variants, providing richer supervision signal
        repeated_indices = valid_indices.repeat(self.num_dn_groups, 1).view(-1)
        repeated_labels = all_gt_labels.repeat(self.num_dn_groups, 1).view(-1).long().to(device)
        repeated_batch_idx = batch_indices.repeat(self.num_dn_groups, 1).view(-1)
        repeated_bboxes = all_gt_bboxes.repeat(self.num_dn_groups, 1).to(device)
        
        # Extract center and scale for noise generation
        # WHY: We only add noise to centers (position), not size/rotation
        # This preserves the object's fundamental properties while corrupting location
        noisy_centers = repeated_bboxes[:, :3].clone()
        bbox_scales = repeated_bboxes[:, 3:6].clone()
        
        # Add noise to centers if noise scale > 0
        # WHY: Conditional noise allows disabling DN during ablation studies
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
        """Add Gaussian noise to bbox centers.
        
        WHY this noise strategy works:
        - Size-proportional noise ensures smaller objects get smaller perturbations
        - Clamping to [0,1] prevents invalid coordinates outside the scene
        - Heavy noise -> background class prevents the model from overfitting to noise
        """
        # Calculate noise variance based on bbox size
        # WHY: Larger objects can tolerate more positional noise than smaller ones
        # The /2 factor makes noise proportional to half the object size
        noise_variance = scales / 2 + self.bbox_noise_trans
        
        # Generate random noise in [-1, 1]  
        # WHY: Uniform distribution in [-1,1] gives symmetric noise around the center
        random_noise = torch.rand_like(centers) * 2 - 1.0
        
        # Apply noise to centers
        # WHY: Multiplicative application scales noise by object size and noise_scale parameter
        centers += torch.mul(random_noise, noise_variance) * self.bbox_noise_scale
        
        # Normalize centers to [0, 1] range
        # WHY: Model expects normalized coordinates, and we need to handle coordinate system
        centers[..., 0:3] = (centers[..., 0:3] - self.pc_range[0:3]) / (
            self.pc_range[3:6] - self.pc_range[0:3]
        )
        centers = centers.clamp(min=0.0, max=1.0)
        
        # Mark heavily noised instances as background class
        # WHY: If noise is too large (L2 norm > split threshold), the instance becomes
        # so corrupted it's better to treat it as background rather than the original class
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
        """Create padded reference points with DN queries.
        
        WHY this padding structure:
        - DN queries come FIRST, then regular queries
        - This ordering ensures DN queries get priority in attention computation
        - Zero padding creates neutral starting points for DN queries
        """
        # Create zero padding for DN queries
        # WHY: Zero initialization provides neutral starting points that don't bias
        # the model towards any particular spatial location
        padding_bbox = torch.zeros(dn_pad_size, 3).to(reference_points.device)
        padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0)
        padded_reference_points = padded_reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Fill in noisy centers if we have GT data
        # WHY: Replace zero padding with actual noisy GT centers at their designated positions
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
        """Create mapping from flattened indices to padded positions.
        
        WHY this complex mapping is needed:
        - GT instances from different batches need to be placed at specific positions
        - Each DN group occupies a distinct block of positions
        - This mapping ensures no overlap between different noise groups
        """
        # Create base mapping for one num_dn_groups copy
        # WHY: This creates [0,1,2,...,n1-1, 0,1,2,...,n2-1, ...] for each batch's GT count
        base_mapping = torch.cat([
            torch.tensor(range(num)) for num in gt_counts_per_batch
        ])
        
        # Extend mapping for all num_dn_groups copies
        # WHY: Each DN group gets its own block of max_gt_per_batch positions
        # This prevents different noise levels from occupying the same positions
        position_mapping = torch.cat([
            base_mapping + max_gt_per_batch * i 
            for i in range(self.num_dn_groups)
        ]).long()
        
        return position_mapping
    
    def _create_attention_masks(
        self,
        dn_pad_size: int,
        max_gt_per_batch: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create attention masks for DN queries.
        
        WHY these specific masking patterns:
        - Prevents DN queries from different groups from seeing each other
        - Maintains the denoising challenge by isolating each noise level
        - Regular queries can't see DN queries to prevent information leakage
        - Temporal extension handles video sequence modeling
        """
        # Basic mask for DN + regular queries
        # WHY: Start with False (can see) everywhere, then selectively mask (True = can't see)
        basic_size = dn_pad_size + self.num_query
        attn_mask = torch.ones(basic_size, basic_size).to(device) < 0
        
        # Regular queries cannot see DN queries
        # WHY: This prevents the regular model queries from "cheating" by looking at
        # the noisy GT information during normal inference-like operation
        attn_mask[dn_pad_size:, :dn_pad_size] = True
        
        # DN queries from different num_dn_groups copies cannot see each other
        # WHY: Each noise group should solve the denoising task independently
        # This prevents easier solutions where one group copies from another
        for i in range(self.num_dn_groups):
            start_idx = max_gt_per_batch * i
            end_idx = max_gt_per_batch * (i + 1)
            
            if i == 0:
                # First copy cannot see later copies
                attn_mask[start_idx:end_idx, end_idx:dn_pad_size] = True
            elif i == self.num_dn_groups - 1:
                # Last copy cannot see earlier copies
                attn_mask[start_idx:end_idx, :start_idx] = True
            else:
                # Middle copies cannot see any other copies
                attn_mask[start_idx:end_idx, end_idx:dn_pad_size] = True
                attn_mask[start_idx:end_idx, :start_idx] = True
        
        # Extend mask for temporal modeling
        # WHY: Video models need additional memory and propagated queries
        # The mask structure must be preserved for these temporal components
        query_size = dn_pad_size + self.num_query + self.num_propagated
        target_size = dn_pad_size + self.num_query + self.memory_len
        temporal_attn_mask = torch.ones(query_size, target_size).to(device) < 0
        
        # Copy basic mask to temporal mask
        # WHY: Preserve the DN isolation rules while extending to temporal dimensions
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
        """Create metadata dictionary for DN training.
        
        WHY these specific keys are needed:
        - 'known_indice': Maps noisy instances back to original GT for loss computation
        - 'batch_idx': Tracks which batch each instance belongs to
        - 'map_known_indice': Maps from flattened space to 2D padded positions
        - 'known_lbs_bboxes': The actual noisy labels and bboxes for supervision
        - 'know_idx': Original validity masks for proper loss masking
        - 'pad_size': Number of DN queries, needed for splitting predictions
        
        This metadata enables the loss computation to properly match predictions
        with their corresponding ground truth targets.
        """
        position_mapping = self._create_position_mapping(gt_counts_per_batch, max_gt_per_batch)
        
        return {
            'known_indice': torch.as_tensor(noisy_data['valid_indices']).long(),
            'batch_idx': torch.as_tensor(batch_indices).long(),
            'map_known_indice': torch.as_tensor(position_mapping).long(),
            'known_lbs_bboxes': (noisy_data['labels'], noisy_data['bboxes']),
            'know_idx': gt_validity_masks,
            'pad_size': dn_pad_size
        }
