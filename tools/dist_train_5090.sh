CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# CUDA_VISIBLE_DEVICES=5

export NCCL_IB_DISABLE=1    # 禁用InfiniBand
export NCCL_P2P_DISABLE=1   # 禁用P2P通信，强制使用PCIe
export NCCL_SHM_DISABLE=1
# 设置MKL线程数
# export MKL_NUM_THREADS=$GPUS
# export OMP_NUM_THREADS=1

# 设置项目根目录和PYTHONPATH
PROJECT_ROOT="/home/yuchen/projects/BEV/BEV_1.2.0/prefusion"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== Training Configuration ==="
echo "Config file: $CONFIG"
echo "GPUs: $GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set (will use all GPUs)}"
echo "Nodes: $NNODES"
echo "Node rank: $NODE_RANK"
echo "Master addr: $MASTER_ADDR"
echo "Port: $PORT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Additional args: ${@:3}"
echo "=============================="

echo "=== GPU Information ==="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
    nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "Using all available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
fi
echo "=============================="
echo ""

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG not found!"
    exit 1
fi

# 检查train.py是否存在
if [ ! -f "tools/train.py" ]; then
    echo "Error: tools/train.py not found!"
    exit 1
fi

# 运行分布式训练
TORCH_COMPILE=1 \
TORCH_COMPILE_MODE=reduce-overhead \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    "$CONFIG" \
    --launcher pytorch \
    "${@:3}"