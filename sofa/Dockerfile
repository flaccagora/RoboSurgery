# Use Nvidia Cuda as base image
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Set non-interactive mode to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    vim \
    unzip \
    libstdc++6 \
    ffmpeg libsm6 libxext6 \
    libopengl0 \
    libegl1-mesa-dev \
    freeglut3-dev \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-dev \ 
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils \
    xvfb \
    libnvidia-gl-470-server \
    && rm -rf /var/lib/apt/lists/*
    
# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Set up Conda environment variables
ENV PATH="/opt/miniconda/bin:$PATH"

RUN conda init bash

# # Create a new Conda environment with Python and other necessary packages
RUN conda create -n sofa python=3.10.15 
RUN conda run -n sofa conda install -c conda-forge gcc=12.1.0 -y

# Activate Conda environment by default in bash sessions
RUN echo "conda activate sofa" >> ~/.bashrc

# Set working directory inside the container
WORKDIR /app

RUN conda run -n sofa bash -c '\
    git clone https://github.com/ScheiklP/sofa_env.git  && \
    cd sofa_env && \
    pip install -e .'

RUN conda run -n sofa bash -c '\
    git clone https://github.com/ScheiklP/sofa_zoo.git  && \
    cd sofa_zoo && \
    pip install -e .'


# Default command: Open an interactive shell
CMD ["/bin/bash"]
