FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/regencode/Vim
RUN pip install -v ./Vim/mamba-1p1p1
RUN pip install -v ./Vim/causal-conv1d
RUN pip install pyunpack patool yacs torchinfo pytorch_lightning ninja packaging monai matplotlib opencv-python scikit-learn transformers timm

ARG CACHEBUST=1
COPY .. .
RUN pip install -e .

ENTRYPOINT ["python", "run.py"]
CMD ["--config", "app/configs/cdmamba-config.yaml"]


