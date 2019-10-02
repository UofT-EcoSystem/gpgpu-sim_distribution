FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && \ 
apt-get install -y --no-install-recommends \ 
    sudo \
    gdb \
    git \
    vim g++ make pkg-config \ 
    libopencv-dev \ 
    libopenblas-dev \ 
    libjemalloc-dev \ 
    python3-dev \ 
    python3-numpy \ 
    python3-pip \ 
    python-pip \
    python3-six \ 
    python3-setuptools \
    build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev \
    python-pmw python-ply libpng12-dev python-matplotlib && \ 
apt-get update && \
apt-get -y --no-install-recommends install apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    wget && \
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main' && \
apt-get update && \
apt-get install -y --no-install-recommends cmake

RUN adduser --home /home/serinatan --shell /bin/bash --uid 3724 --gecos '' --disabled-password serinatan  
RUN adduser serinatan sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers 
RUN rm -rf /var/lib/apt/lists/* # installation 
