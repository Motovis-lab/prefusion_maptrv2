FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx build-essential wget git curl libsm6 libxrender1 \
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

# install requirements
# RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
COPY requirements.txt /requirements.txt
COPY mim-requirements.txt /mim-requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir flash-attn==0.2.2
RUN pip install --no-cache-dir openmim
RUN mim install --no-cache-dir -r /mim-requirements.txt

# RUN mv /etc/apt/sources.list.bak /etc/apt/sources.list

WORKDIR /workspace

CMD [ "/bin/bash"  ]

