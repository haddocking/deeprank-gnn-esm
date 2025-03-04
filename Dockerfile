# ==============================================================================
FROM continuumio/anaconda3:2023.03-1 AS base

# Use an argument to select either cpu or gpu environment
ARG ENV_TYPE=cpu

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential=12.9 \
  git=1:2.30.2-1+deb11u2 \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy app
WORKDIR /opt/deeprank-gnn-esm
COPY . .

# Setup environment using the dynamically set environment file
RUN conda env create -f environment-${ENV_TYPE}.yml -n env

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN source activate env

# Install deeprank-gnn-esm
RUN pip install .

# Setup entrypoint
WORKDIR /data
ENTRYPOINT [ "deeprank-gnn-esm" ]

# ==============================================================================
