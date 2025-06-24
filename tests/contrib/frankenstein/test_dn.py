"""
Pytest tests for NoisyInstanceGenerator and prepare_for_dn functionality.

This test suite verifies that the refactored NoisyInstanceGenerator produces
identical outputs to the original prepare_for_dn implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import pytest
import torch
import torch.nn as nn
from typing import List

# Import the new implementation
from contrib.frankenstein.noisy_instance_generator import NoisyInstanceGenerator


class OriginalDNImplementation(nn.Module):
    """
    Original prepare_for_dn implementation for comparison testing.
    
    This class contains the exact original logic from FrankenStreamPETRHead.prepare_for_dn
    to serve as ground truth for testing the refactored implementation.
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
        self.register_buffer('pc_range', torch.tensor(pc_range))
        
        # For DN training
        self.training = True
        self.with_dn = True
    
    def prepare_for_dn(self, batch_size, reference_points, gt_bboxes_3d, gt_labels):
        """Original prepare_for_dn implementation - EXACT COPY from franken_head.py"""
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
            known_labels = gt_labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
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
                known_labels[mask] = self.num_classes # discussion: https://github.com/exiawsh/StreamPETR/issues/233

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
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
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
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
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict


class TestNoisyInstanceGenerator:
    """Test suite for NoisyInstanceGenerator."""
    
    @pytest.fixture
    def device(self):
        """Fixture to provide device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def default_config(self):
        """Fixture providing default configuration."""
        return {
            'num_classes': 10,
            'num_query': 100,
            'num_propagated': 256,
            'memory_len': 1024,
            'num_dn_groups': 5,
            'bbox_noise_scale': 0.4,
            'bbox_noise_trans': 0.0,
            'split': 0.5,
            'pc_range': [-65, -65, -8.0, 65, 65, 8.0]
        }
    
    @pytest.fixture
    def sample_data(self, device):
        """Fixture providing sample input data."""
        batch_size = 2
        
        # Create reference points
        reference_points = torch.rand(100, 3, device=device)
        
        # Create GT bboxes and labels for each batch
        gt_bboxes_3d = [
            torch.rand(3, 10, device=device) * 50,  # 3 boxes with 10 dims each
            torch.rand(2, 10, device=device) * 50   # 2 boxes with 10 dims each
        ]
        
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),  # 3 labels
            torch.randint(0, 10, (2,), device=device)   # 2 labels
        ]
        
        return {
            'batch_size': batch_size,
            'reference_points': reference_points,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': gt_labels
        }
    
    def test_basic_functionality(self, default_config, sample_data, device):
        """Test basic functionality of NoisyInstanceGenerator."""
        generator = NoisyInstanceGenerator(**default_config).to(device)
        
        result = generator.generate_noisy_instances(
            sample_data['batch_size'],
            sample_data['reference_points'],
            sample_data['gt_bboxes_3d'],
            sample_data['gt_labels'],
            training=True,
            with_dn=True
        )
        
        padded_reference_points, attn_mask, mask_dict = result
        
        # Basic shape checks
        assert padded_reference_points is not None
        assert attn_mask is not None
        assert mask_dict is not None
        
        # Check shapes
        expected_dn_size = max(len(gt) for gt in sample_data['gt_bboxes_3d']) * default_config['num_dn_groups']
        expected_total_queries = expected_dn_size + default_config['num_query']
        
        assert padded_reference_points.shape == (sample_data['batch_size'], expected_total_queries, 3)
        assert mask_dict['pad_size'] == expected_dn_size
    
    def test_non_dn_mode(self, default_config, sample_data, device):
        """Test behavior when DN is disabled."""
        generator = NoisyInstanceGenerator(**default_config).to(device)
        
        result = generator.generate_noisy_instances(
            sample_data['batch_size'],
            sample_data['reference_points'],
            sample_data['gt_bboxes_3d'],
            sample_data['gt_labels'],
            training=False,  # DN disabled
            with_dn=True
        )
        
        padded_reference_points, attn_mask, mask_dict = result
        
        assert padded_reference_points.shape == (sample_data['batch_size'], default_config['num_query'], 3)
        assert attn_mask is None
        assert mask_dict is None
    
    def test_empty_gt_data(self, default_config, device):
        """Test behavior with empty GT data."""
        generator = NoisyInstanceGenerator(**default_config).to(device)
        
        reference_points = torch.rand(100, 3, device=device)
        empty_gt_bboxes = [torch.empty(0, 10, device=device)]
        empty_gt_labels = [torch.empty(0, dtype=torch.long, device=device)]
        
        result = generator.generate_noisy_instances(
            1,  # batch_size
            reference_points,
            empty_gt_bboxes,
            empty_gt_labels,
            training=True,
            with_dn=True
        )
        
        padded_reference_points, attn_mask, mask_dict = result
        
        # Should still work with empty GT
        assert padded_reference_points.shape == (1, 100, 3)  # Only original queries
        assert mask_dict['pad_size'] == 0

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_equivalence_with_original(self, default_config, sample_data, device, seed):
        """
        Test that NoisyInstanceGenerator produces identical results to original implementation.

        This is the key test that verifies our refactoring is correct.
        """
        torch.manual_seed(seed)

        # Create both implementations with identical configuration
        new_generator = NoisyInstanceGenerator(**default_config).to(device)
        
        # Convert config for original implementation (uses 'scalar' instead of 'num_dn_groups')
        original_config = default_config.copy()
        original_config['scalar'] = original_config.pop('num_dn_groups')
        original_impl = OriginalDNImplementation(**original_config).to(device)
        
        # Test with DN enabled
        torch.manual_seed(seed)  # Reset seed for reproducible random operations
        new_result = new_generator.generate_noisy_instances(
            sample_data['batch_size'],
            sample_data['reference_points'],
            sample_data['gt_bboxes_3d'],
            sample_data['gt_labels'],
            training=True,
            with_dn=True
        )
        
        torch.manual_seed(seed)  # Reset seed again for original implementation
        original_result = original_impl.prepare_for_dn(
            sample_data['batch_size'],
            sample_data['reference_points'],
            sample_data['gt_bboxes_3d'],
            sample_data['gt_labels']
        )
        
        # Compare results
        new_padded_ref, new_attn_mask, new_mask_dict = new_result
        orig_padded_ref, orig_attn_mask, orig_mask_dict = original_result
        
        # Compare padded reference points
        assert torch.allclose(new_padded_ref, orig_padded_ref, atol=1e-6), \
            f"Padded reference points differ. New shape: {new_padded_ref.shape}, Original shape: {orig_padded_ref.shape}"
        
        # Compare attention masks
        assert torch.equal(new_attn_mask, orig_attn_mask), \
            f"Attention masks differ. New shape: {new_attn_mask.shape}, Original shape: {orig_attn_mask.shape}"
        
        # Compare mask dictionaries
        assert new_mask_dict['pad_size'] == orig_mask_dict['pad_size']
        assert torch.equal(new_mask_dict['known_indice'], orig_mask_dict['known_indice'])
        assert torch.equal(new_mask_dict['batch_idx'], orig_mask_dict['batch_idx'])
        assert torch.equal(new_mask_dict['map_known_indice'], orig_mask_dict['map_known_indice'])
        
        # Compare known labels and bboxes
        new_labels, new_bboxes = new_mask_dict['known_lbs_bboxes']
        orig_labels, orig_bboxes = orig_mask_dict['known_lbs_bboxes']
        
        assert torch.equal(new_labels, orig_labels), "Known labels differ"
        assert torch.allclose(new_bboxes, orig_bboxes, atol=1e-6), "Known bboxes differ"
    
    def test_different_configurations(self, device):
        """Test with different configurations to ensure robustness."""
        configs = [
            # Standard config
            {
                'num_classes': 10, 'num_query': 100, 'num_dn_groups': 5,
                'bbox_noise_scale': 0.4, 'split': 0.5
            },
            # No noise config
            {
                'num_classes': 5, 'num_query': 50, 'num_dn_groups': 3,
                'bbox_noise_scale': 0.0, 'split': 0.5
            },
            # High noise config
            {
                'num_classes': 20, 'num_query': 200, 'num_dn_groups': 10,
                'bbox_noise_scale': 1.0, 'split': 0.8
            }
        ]
        
        for config in configs:
            full_config = {
                'num_propagated': 256,
                'memory_len': 1024,
                'bbox_noise_trans': 0.0,
                'pc_range': [-65, -65, -8.0, 65, 65, 8.0],
                **config
            }
            
            generator = NoisyInstanceGenerator(**full_config).to(device)
            
            # Create sample data
            reference_points = torch.rand(full_config['num_query'], 3, device=device)
            gt_bboxes_3d = [torch.rand(2, 10, device=device) * 50]
            gt_labels = [torch.randint(0, full_config['num_classes'], (2,), device=device)]
            
            result = generator.generate_noisy_instances(
                1, reference_points, gt_bboxes_3d, gt_labels,
                training=True, with_dn=True
            )
            
            padded_reference_points, attn_mask, mask_dict = result
            
            # Verify basic properties
            assert padded_reference_points is not None
            assert attn_mask is not None
            assert mask_dict is not None
            assert mask_dict['pad_size'] == 2 * full_config['num_dn_groups']
    
    def test_attention_mask_properties(self, default_config, sample_data, device):
        """Test specific properties of attention masks."""
        generator = NoisyInstanceGenerator(**default_config).to(device)
        
        result = generator.generate_noisy_instances(
            sample_data['batch_size'],
            sample_data['reference_points'],
            sample_data['gt_bboxes_3d'],
            sample_data['gt_labels'],
            training=True,
            with_dn=True
        )
        
        _, attn_mask, mask_dict = result
        pad_size = mask_dict['pad_size']
        
        # Regular queries should not see DN queries
        regular_query_start = pad_size
        assert torch.all(attn_mask[regular_query_start:, :pad_size]), \
            "Regular queries should not see DN queries"
        
        # DN queries should have specific masking patterns
        max_gt_per_batch = max(len(gt) for gt in sample_data['gt_bboxes_3d'])
        
        for i in range(default_config['num_dn_groups']):
            start = i * max_gt_per_batch
            end = (i + 1) * max_gt_per_batch
            
            if i < default_config['num_dn_groups'] - 1:
                # Should not see later copies
                later_start = (i + 1) * max_gt_per_batch
                assert torch.all(attn_mask[start:end, later_start:pad_size]), \
                    f"Group {i} should not see later copies"
            
            if i > 0:
                # Should not see earlier copies
                assert torch.all(attn_mask[start:end, :start]), \
                    f"Group {i} should not see earlier copies"


def print_ground_truth_outputs():
    """
    Helper function to print ground truth outputs for manual verification.
    
    This function generates outputs using the original implementation with
    fixed seeds to create reproducible ground truth data.
    """
    print("=== GROUND TRUTH OUTPUTS FOR MANUAL VERIFICATION ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'num_classes': 10,
        'num_query': 100,
        'num_propagated': 256,
        'memory_len': 1024,
        'scalar': 5,
        'bbox_noise_scale': 0.4,
        'bbox_noise_trans': 0.0,
        'split': 0.5,
        'pc_range': [-65, -65, -8.0, 65, 65, 8.0]
    }
    
    # Create sample data
    torch.manual_seed(42)
    batch_size = 2
    reference_points = torch.rand(100, 3, device=device)
    gt_bboxes_3d = [
        torch.rand(3, 10, device=device) * 50,
        torch.rand(2, 10, device=device) * 50
    ]
    gt_labels = [
        torch.randint(0, 10, (3,), device=device),
        torch.randint(0, 10, (2,), device=device)
    ]
    
    # Generate output with original implementation
    original_impl = OriginalDNImplementation(**config).to(device)
    torch.manual_seed(42)  # Reset for reproducible random operations
    
    padded_ref, attn_mask, mask_dict = original_impl.prepare_for_dn(
        batch_size, reference_points, gt_bboxes_3d, gt_labels
    )
    
    print(f"Input GT counts: {[len(gt) for gt in gt_bboxes_3d]}")
    print(f"Padded reference points shape: {padded_ref.shape}")
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Pad size: {mask_dict['pad_size']}")
    print(f"Known indices shape: {mask_dict['known_indice'].shape}")
    print(f"Batch indices shape: {mask_dict['batch_idx'].shape}")
    print(f"Map known indices shape: {mask_dict['map_known_indice'].shape}")
    
    labels, bboxes = mask_dict['known_lbs_bboxes']
    print(f"Known labels shape: {labels.shape}")
    print(f"Known bboxes shape: {bboxes.shape}")
    
    print("\n=== SAMPLE VALUES ===")
    print(f"First few padded reference points:\n{padded_ref[0, :5]}")
    print(f"First few known labels: {labels[:10]}")
    print(f"First few map indices: {mask_dict['map_known_indice'][:10]}")


if __name__ == "__main__":
    # Run ground truth output printing when executed directly
    print_ground_truth_outputs()
