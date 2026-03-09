FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# 关键：base image 自带 torch 2.4.1+cu124（CUDA 专用版），绝不能让 pip 升级它
# 先用 --no-deps 装 diffusers 和 accelerate，避免它们的依赖链拉新 torch
COPY requirements_runpod.txt .
RUN pip install --no-cache-dir --no-deps "diffusers>=0.36.0" accelerate optimum-quanto && \
    pip install --no-cache-dir runpod Pillow "transformers>=4.49.0,<5.0.0" tqdm sentencepiece && \
    pip install --no-cache-dir --force-reinstall "transformers>=4.49.0,<5.0.0"

# 模型不打入镜像，运行时从 Network Volume 或 HuggingFace 加载
# Network Volume 路径: /runpod-volume/models/FireRed-Image-Edit-1.1
# 若 volume 不存在则自动从 HuggingFace 下载（冷启动较慢）

COPY handler.py .

CMD ["python", "-u", "handler.py"]
