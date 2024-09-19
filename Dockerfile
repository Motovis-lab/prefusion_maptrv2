FROM python:3.10

# RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
# RUN echo "deb https://mirrors.aliyun.com/debian/ bullseye main non-free contrib" > /etc/apt/sources.list
# RUN echo "deb-src https://mirrors.aliyun.com/debian/ bullseye main non-free contrib" >> /etc/apt/sources.list
# RUN echo "deb https://mirrors.aliyun.com/debian-security/ bullseye-security main" >> /etc/apt/sources.list
# RUN echo "deb-src https://mirrors.aliyun.com/debian-security/ bullseye-security main" >> /etc/apt/sources.list
# RUN echo "deb https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib" >> /etc/apt/sources.list
# RUN echo "deb-src https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib" >> /etc/apt/sources.list
# RUN echo "deb https://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib" >> /etc/apt/sources.list
# RUN echo "deb-src https://mirrors.aliyun.com/debian/ bullseye-backports main non-free contrib" >> /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

COPY requirements.txt /requirements.txt
COPY mim-requirements.txt /mim-requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir openmim
RUN mim install --no-cache-dir -r /mim-requirements.txt

# RUN mv /etc/apt/sources.list.bak /etc/apt/sources.list

CMD ["python3"]