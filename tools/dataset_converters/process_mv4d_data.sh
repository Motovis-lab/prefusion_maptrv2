#!/bin/bash

# 检查是否提供了目录参数
if [ $# -eq 0 ]; then
    echo "请提供一个目录作为参数"
    exit 1
fi

# 获取提供的目录路径
directory="$1"

# 检查目录是否存在
if [ ! -d "$directory" ]; then
    echo "错误：提供的路径 '$directory' 不是一个有效的目录"
    exit 1
fi


# 获取可用的 CPU 核心数
num_cores=$(nproc)

# 定义一个函数来处理每个文件夹
process_folder() {
    folder_name="$1"
    echo "处理文件夹: $folder_name"
    python tools/dataset_converters/gene_info_4d.py "$folder_name"
    python tools/dataset_converters/shutil_file.py "$folder_name"
}

# 计数器，用于跟踪当前运行的作业数
job_count=0

# 查找以 2023 开头的文件夹并处理
find "$directory" -maxdepth 1 -type d -name "2023*" | while read -r folder; do
    folder_name=$(basename "$folder")
    
    # 在后台运行处理函数
    process_folder "$folder_name" &
    
    # 增加作业计数
    ((job_count++))
    
    # 如果达到核心数量，等待一个作业完成
    if [ $job_count -eq $num_cores ]; then
        wait -n
        ((job_count--))
    fi
done

# 等待所有剩余的后台作业完成
wait

echo "所有匹配的文件夹都已处理完毕"
