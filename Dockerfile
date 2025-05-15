FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx build-essential wget git curl libsm6 libxrender1 vim htop iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install anaconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh --output conda_installer.sh
RUN /bin/bash conda_installer.sh -b -p /opt/conda \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda clean -afy
RUN rm conda_installer.sh

ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda

# conda install critical libraries
COPY .condarc /root/.condarc
RUN conda install -y numba=0.60.0 numpy=1.26.4 scipy=1.13.1 pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# pip install other libraries
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir copious==0.1.23 easydict==1.13 nuscenes-devkit opencv-python-headless virtual-camera==0.0.4.3 scikit-learn==1.4.2 open3d==0.17.0 transformers==4.45.2 pypcd-imp==0.1.5 mmengine==0.10.5 mmdet3d==1.4.0 mmdet==3.2.0 pytest==8.3.3 redis==5.2.0 s3cmd==2.3.0 s5cmd pytest-cov loguru tqdm
RUN pip install --no-cache-dir ninja

# If not use docker build, we need to temporarily put cuda (of correct version) to /usr/loca/cuda or conda install cuda-toolkit cuda-cudart cuda-cccl libcublas libcusparse libcusolver
# RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3 -v
COPY flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation /flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -v
RUN rm -rf /flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install mmcv
RUN pip install --no-cache-dir openmim
COPY mim-requirements.txt /mim-requirements.txt
RUN mim install --no-cache-dir -r /mim-requirements.txt

# RUN mv /etc/apt/sources.list.bak /etc/apt/sources.list

# install libgllib2.0 and set TimeZone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt-get update \
   && apt-get install -y libglib2.0-0 libxext6 tzdata  \
   && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
   && echo $TZ > /etc/timezone \
   && dpkg-reconfigure -f noninteractive tzdata \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# install pytorch3d
# RUN pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121
COPY ./pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl
RUN rm -rf /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl

# install extra system utils

# install extra python packages
RUN pip install --no-cache-dir spconv-cu121==2.3.7
RUN pip install --no-cache-dir onnx==1.17.0
RUN pip install --no-cache-dir onnxruntime==1.20.1
RUN pip install --no-cache-dir copious==0.1.25
RUN pip install --no-cache-dir mtv4d==0.1.4
RUN pip install --no-cache-dir polars==1.28.0

WORKDIR /workspace

CMD [ "/bin/bash"  ]

