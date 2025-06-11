# Checkpoint Conversion Script

## Overview

This directory contains a script to convert checkpoints trained by the prefusion framework to the format compatible with the original StreamPETR implementation.

## Usage

### Basic Conversion

```bash
python convert_to_original_ckpt.py --input path/to/prefusion/checkpoint.pth
```

### Conversion with Validation

```bash
python convert_to_original_ckpt.py \
  --input path/to/prefusion/checkpoint.pth \
  --reference path/to/original/streampetr/checkpoint.pth \
  --verbose
```

### Full Options

```bash
python convert_to_original_ckpt.py \
  --input path/to/prefusion/checkpoint.pth \
  --output path/to/converted/checkpoint.pth \
  --reference path/to/original/streampetr/checkpoint.pth \
  --verbose
```

## Key Mappings

The script applies the following parameter name mappings:

| Prefusion Framework | Original StreamPETR |
|-------------------|-------------------|
| `box_head.*` | `pts_bbox_head.*` |
| `roi_head.*` | `img_roi_head.*` |
| `img_backbone.*` | `img_backbone.*` (unchanged) |
| `img_neck.*` | `img_neck.*` (unchanged) |

## Arguments

- `--input, -i`: Path to input prefusion checkpoint (required)
- `--output, -o`: Path to output converted checkpoint (optional, auto-generated if not provided)
- `--reference, -r`: Path to reference original checkpoint for validation (optional)
- `--verbose, -v`: Enable verbose output for detailed analysis

## Example

```bash
# Convert prefusion checkpoint with validation
python convert_to_original_ckpt.py \
  --input /path/to/prefusion/ckpts/stream_petr_epoch_24.pth \
  --reference /path/to/StreamPETR/ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth \
  --verbose
```

This will:
1. Load the prefusion checkpoint
2. Apply the key mappings to convert parameter names
3. Validate against the reference checkpoint (if provided)
4. Save the converted checkpoint with the original StreamPETR format

## Validation

When a reference checkpoint is provided, the script will:
- Compare the number of parameters
- Check that all key names match
- Verify that parameter shapes are identical
- Report any mismatches or issues

## Output

The converted checkpoint will have the same structure as the original StreamPETR checkpoints and can be used directly with the original implementation.
