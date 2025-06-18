#!/bin/bash

# nusc_infer_and_eval.sh
# Script to run inference followed by evaluation for NuScenes dataset
# Usage: ./nusc_infer_and_eval.sh <config_path> <checkpoint_path> <work_dir> <nusc_data_root> [gpu_id] [eval_set] [scene_names...]

set -e  # Exit on any error

# Check if required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <config_path> <checkpoint_path> <work_dir> <nusc_data_root> [gpu_id] [eval_set] [scene_names...]"
    echo ""
    echo "Arguments:"
    echo "  config_path     - Path to config file (e.g., contrib/petr/configs/streampetr_nusc_r50_704x256.py)"
    echo "  checkpoint_path - Path to checkpoint file (e.g., /path/to/checkpoint.pth)"
    echo "  work_dir        - Work directory for output (e.g., eval_results/model_name)"
    echo "  nusc_data_root  - Path to NuScenes dataset root (e.g., /ssd4/datasets/nuscenes)"
    echo "  gpu_id          - GPU ID to use (default: 0)"
    echo "  eval_set        - Evaluation set (default: val)"
    echo "  scene_names...  - Optional scene names to evaluate (e.g., scene-0001 scene-0002)"
    echo ""
    echo "Example:"
    echo "  $0 contrib/petr/configs/streampetr_nusc_r50_704x256.py \\"
    echo "     /home/user/ckpts/checkpoint.pth \\"
    echo "     eval_results/my_model \\"
    echo "     /ssd4/datasets/nuscenes \\"
    echo "     7 \\"
    echo "     val \\"
    echo "     scene-0001 scene-0002"
    exit 1
fi

# Parse arguments
CONFIG_PATH=$1
CHECKPOINT_PATH=$2
WORK_DIR=$3
NUSC_DATA_ROOT=$4
GPU_ID=${5:-0}  # Default to GPU 0 if not specified
EVAL_SET=${6:-val}  # Default to val set if not specified
shift 6  # Remove the first 6 arguments
SCENE_NAMES=("$@")  # Remaining arguments are scene names

# Validate that we're in the project root
if [ ! -f "tools/infer.py" ] || [ ! -f "tools/evaluator/evaluate_nusc_certain_scene.py" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Validate input files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$NUSC_DATA_ROOT" ]; then
    echo "Error: NuScenes data root not found: $NUSC_DATA_ROOT"
    exit 1
fi

echo "============================================"
echo "NuScenes Inference and Evaluation Script"
echo "============================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Work directory: $WORK_DIR"
echo "NuScenes data root: $NUSC_DATA_ROOT"
echo "GPU ID: $GPU_ID"
echo "Evaluation set: $EVAL_SET"
if [ ${#SCENE_NAMES[@]} -gt 0 ]; then
    echo "Scene names: ${SCENE_NAMES[*]}"
else
    echo "Scene names: all scenes in $EVAL_SET set"
fi
echo "============================================"

# Step 1: Run inference
echo ""
echo "Step 1: Running inference..."
echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=\$(pwd) python tools/infer.py $CONFIG_PATH --cfg-options load_from=$CHECKPOINT_PATH work_dir=$WORK_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH=$(pwd) python tools/infer.py \
    "$CONFIG_PATH" \
    --cfg-options \
    load_from="$CHECKPOINT_PATH" \
    work_dir="$WORK_DIR"

# Check if inference was successful
if [ $? -ne 0 ]; then
    echo "Error: Inference failed!"
    exit 1
fi

# Check if the inference results file exists
RESULTS_FILE="$WORK_DIR/nusc_det_results.json"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Inference results file not found: $RESULTS_FILE"
    exit 1
fi

echo "Inference completed successfully!"

# Step 2: Run evaluation
echo ""
echo "Step 2: Running evaluation..."

# Build the evaluation command
EVAL_CMD="python tools/evaluator/evaluate_nusc_certain_scene.py \
    --nusc-data-root \"$NUSC_DATA_ROOT\" \
    --nusc-eval-set \"$EVAL_SET\" \
    --model-infer-results \"$RESULTS_FILE\" \
    --output-dir \"$WORK_DIR\""

# Add scene names if provided
if [ ${#SCENE_NAMES[@]} -gt 0 ]; then
    EVAL_CMD="$EVAL_CMD --scene-names ${SCENE_NAMES[*]}"
fi

echo "Command: $EVAL_CMD"

# Execute the evaluation command
eval $EVAL_CMD

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "Pipeline completed successfully!"
echo "Results saved in: $WORK_DIR"
echo "============================================"