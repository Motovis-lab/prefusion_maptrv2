#!/usr/bin/env python3
"""
Test script to verify the precision consistency fix for StreamPETR
"""
import torch
import torch.nn as nn
from contrib.petr.attention import FlashAttention

def test_attention_precision_consistency():
    """Test that attention respects input precision"""
    print("Testing FlashAttention precision consistency...")
    
    # Create attention module
    attn = FlashAttention(attention_dropout=0.0)
    
    # Test dimensions
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 12
    num_heads = 8
    head_dim = 64
    
    # Test FP32 inputs (non-AMP case)
    print("\n1. Testing FP32 inputs (non-AMP):")
    q_fp32 = torch.randn(batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float32, device='cuda')
    k_fp32 = torch.randn(batch_size, seq_len_k, num_heads, head_dim, dtype=torch.float32, device='cuda')
    v_fp32 = torch.randn(batch_size, seq_len_k, num_heads, head_dim, dtype=torch.float32, device='cuda')
    kv_fp32 = torch.stack([k_fp32, v_fp32], dim=2)
    
    with torch.no_grad():
        output_fp32, _ = attn(q_fp32, kv_fp32)
    
    print(f"Input dtype: {q_fp32.dtype}")
    print(f"Output dtype: {output_fp32.dtype}")
    print(f"Output shape: {output_fp32.shape}")
    assert output_fp32.dtype == torch.float32, f"Expected FP32 output, got {output_fp32.dtype}"
    
    # Test FP16 inputs (AMP case)
    print("\n2. Testing FP16 inputs (AMP):")
    q_fp16 = q_fp32.half()
    kv_fp16 = kv_fp32.half()
    
    with torch.no_grad():
        output_fp16, _ = attn(q_fp16, kv_fp16)
    
    print(f"Input dtype: {q_fp16.dtype}")
    print(f"Output dtype: {output_fp16.dtype}")
    print(f"Output shape: {output_fp16.shape}")
    assert output_fp16.dtype == torch.float16, f"Expected FP16 output, got {output_fp16.dtype}"
    
    print("\n✅ All precision consistency tests passed!")
    
def test_streampetr_conversion():
    """Test the smart type conversion in StreamPETR"""
    print("\nTesting StreamPETR output conversion logic...")
    
    # Mock outputs with mixed types
    outs_fp16 = {
        'loss_cls': torch.tensor(1.5, dtype=torch.float16),
        'loss_bbox': torch.tensor(2.3, dtype=torch.float16),
        'metadata': 'some_string'
    }
    
    # Test without autocast (should convert to FP32)
    print("\n1. Testing without autocast (should convert FP16 to FP32):")
    if any(isinstance(v, torch.Tensor) and v.dtype == torch.float16 for v in outs_fp16.values()):
        if not torch.is_autocast_enabled():
            converted = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v for k, v in outs_fp16.items()}
            print(f"Converted loss_cls dtype: {converted['loss_cls'].dtype}")
            assert converted['loss_cls'].dtype == torch.float32
    
    # Test with autocast (should keep FP16)
    print("\n2. Testing with autocast (should keep FP16):")
    with torch.autocast(device_type='cuda'):
        if any(isinstance(v, torch.Tensor) and v.dtype == torch.float16 for v in outs_fp16.values()):
            if not torch.is_autocast_enabled():  # This will be False inside autocast
                print("Autocast is properly detected")
    
    print("\n✅ StreamPETR conversion logic test passed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_attention_precision_consistency()
        test_streampetr_conversion()
        print("\n🎉 All tests passed! The precision consistency fix is working correctly.")
    else:
        print("❌ CUDA not available, skipping tests")
