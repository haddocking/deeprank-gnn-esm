# ==============================================================================
ARG CUDA=12.8.0
FROM nvidia/cuda:${CUDA}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  git \
  wget \
  bzip2 \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install anaconda3
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm /tmp/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# Copy deeprank-gnn-esm
WORKDIR /opt/deeprank-gnn-esm
COPY . .

# Setup environment and install
RUN conda tos accept &&\
  conda env create -f environment-gpu.yml -n env && \
  conda clean -afy
ENV PYTHONPATH="/opt/deeprank-gnn-esm/src:$PYTHONPATH"

# Define the entrypoint
ENTRYPOINT ["conda", "run", "-n", "env", "python", "/opt/deeprank-gnn-esm/src/deeprank_gnn/predict.py"]

# ==============================================================================
