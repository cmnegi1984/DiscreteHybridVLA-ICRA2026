FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip git curl && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir \
    torch==2.2.0 torchvision==0.17.0 \
    transformers==4.40.0 \
    numpy pandas boto3 pyarrow \
    opencv-python-headless Pillow
WORKDIR /workspace
COPY . /workspace/
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
CMD ["bash"]
