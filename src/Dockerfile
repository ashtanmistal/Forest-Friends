FROM nvcr.io/nvidia/rapidsai/base:24.04-cuda12.2-py3.11

# important: use the following run options when building the image if not using docker-compose.yml
# --gpus all --rm -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake  \
    wget \
    git  \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/* 


WORKDIR /workspace/gpu_hdbscan

RUN mkdir -p build && cd build \
  && cmake .. && \
  make


ENTRYPOINT [ "./build/ForestFriends"]  # run the ForestFriends executable