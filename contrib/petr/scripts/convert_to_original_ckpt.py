import torch
import argparse
import os

def convert_prefusion_to_original(state_dict):
    """
    Convert prefusion checkpoint state_dict to original StreamPETR format.
    
    Key mappings:
    - box_head -> pts_bbox_head
    - roi_head -> img_roi_head  
    - img_backbone -> img_backbone (unchanged)
    - img_neck -> img_neck (unchanged)
    """
    converted_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Apply conversions
        if key.startswith('box_head.'):
            new_key = key.replace('box_head.', 'pts_bbox_head.')
        elif key.startswith('roi_head.'):
            new_key = key.replace('roi_head.', 'img_roi_head.')
        # img_backbone and img_neck remain unchanged
        
        converted_state_dict[new_key] = value
    
    return converted_state_dict

def main():
    parser = argparse.ArgumentParser(description='Convert Prefusion checkpoint to original StreamPETR format')
    parser.add_argument('--input', '-i', required=True, help='Path to input prefusion checkpoint')
    parser.add_argument('--output', '-o', help='Path to output converted checkpoint (optional)')
    parser.add_argument('--reference', '-r', help='Path to reference original checkpoint for validation (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output for analysis')
    
    args = parser.parse_args()
    
    # Load prefusion checkpoint
    print(f"Loading prefusion checkpoint from: {args.input}")
    prefusion_ckpt = torch.load(args.input, weights_only=False, map_location='cpu')
    
    # Load reference checkpoint if provided
    original_ckpt = None
    if args.reference:
        print(f"Loading reference checkpoint from: {args.reference}")
        original_ckpt = torch.load(args.reference, weights_only=False, map_location='cpu')

    if args.verbose:
        print("\n=== PREFUSION CHECKPOINT ANALYSIS ===")
        print("Top-level keys:", list(prefusion_ckpt.keys()))
        
        if 'state_dict' in prefusion_ckpt:
            state_dict = prefusion_ckpt['state_dict']
        else:
            state_dict = prefusion_ckpt
            
        # Group keys by component
        prefusion_groups = {}
        for key in state_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in prefusion_groups:
                prefusion_groups[prefix] = []
            prefusion_groups[prefix].append(key)
            
        print("Key prefixes:")
        for prefix, keys in prefusion_groups.items():
            print(f"  {prefix}: {len(keys)} keys")
    
    # Convert the checkpoint
    if 'state_dict' in prefusion_ckpt:
        converted_state_dict = convert_prefusion_to_original(prefusion_ckpt['state_dict'])
    else:
        converted_state_dict = convert_prefusion_to_original(prefusion_ckpt)
    
    # Create new checkpoint in original format
    converted_ckpt = {
        'state_dict': converted_state_dict
    }
    
    print("\n=== CONVERSION RESULTS ===")
    print(f"Total parameters converted: {len(converted_state_dict)}")
    
    # Validation against reference if provided
    if original_ckpt:
        original_state_dict = original_ckpt['state_dict'] if 'state_dict' in original_ckpt else original_ckpt
        
        print(f"Reference checkpoint parameters: {len(original_state_dict)}")
        
        # Check key matching
        converted_keys = set(converted_state_dict.keys())
        original_keys = set(original_state_dict.keys())
        
        missing_keys = original_keys - converted_keys
        extra_keys = converted_keys - original_keys
        
        if missing_keys:
            print(f"⚠️  Missing keys in conversion: {len(missing_keys)}")
            if args.verbose:
                for key in list(missing_keys)[:5]:
                    print(f"    - {key}")
                if len(missing_keys) > 5:
                    print(f"    ... and {len(missing_keys)-5} more")
        
        if extra_keys:
            print(f"⚠️  Extra keys in conversion: {len(extra_keys)}")
            if args.verbose:
                for key in list(extra_keys)[:5]:
                    print(f"    + {key}")
                if len(extra_keys) > 5:
                    print(f"    ... and {len(extra_keys)-5} more")
        
        if not missing_keys and not extra_keys:
            print("✅ All keys match perfectly!")
            
            # Check shapes for common keys
            shape_mismatches = 0
            for key in converted_keys:
                if key in original_keys:
                    converted_shape = converted_state_dict[key].shape
                    original_shape = original_state_dict[key].shape
                    if converted_shape != original_shape:
                        shape_mismatches += 1
                        if args.verbose:
                            print(f"⚠️  Shape mismatch for {key}: {converted_shape} vs {original_shape}")
            
            if shape_mismatches == 0:
                print("✅ All parameter shapes match perfectly!")
            else:
                print(f"⚠️  Found {shape_mismatches} shape mismatches")
    
    # Save converted checkpoint
    if args.output:
        output_path = args.output
    else:
        # Generate output path based on input path
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(input_dir, f"{input_name}_converted_to_original.pth")
    
    torch.save(converted_ckpt, output_path)
    print(f"✅ Converted checkpoint saved to: {output_path}")
    
    print("\n=== CONVERSION SUMMARY ===")
    print("Successfully converted prefusion checkpoint to original StreamPETR format!")
    print("Key mappings applied:")
    print("  - box_head.* -> pts_bbox_head.*")
    print("  - roi_head.* -> img_roi_head.*") 
    print("  - img_backbone.* -> img_backbone.* (unchanged)")
    print("  - img_neck.* -> img_neck.* (unchanged)")

if __name__ == "__main__":
    main()
