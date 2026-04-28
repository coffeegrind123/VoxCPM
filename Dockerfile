FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM=0.1.0

RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir funasr modelscope hf-transfer

COPY app.py preload.py ./
COPY assets/ ./assets/
COPY conf/ ./conf/

ENV HF_HOME=/app/.cache/huggingface
ENV MODELSCOPE_CACHE=/app/.cache/modelscope
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 8808

CMD ["python", "-u", "preload.py"]
