FROM python:3.11-slim-bookworm

WORKDIR /app

RUN (sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null \
  || sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list)

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 只设 index-url，**不要加 trusted-host**：index-url 本就是 https，而 `pip config set
# global.trusted-host` 会对该源关闭证书校验，且写进镜像的 pip.conf 长期生效（后续在容器里
# 手动 pip install 也一样不校验）。纯粹减防护、零收益。
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 1) CPU 版 PyTorch（避开 NVIDIA 全家桶）
#    放在 COPY requirements.txt 之前，作独立缓存层：torch 版本固定、不依赖 requirements 内容，
#    这样以后改 requirements.txt 只失效下面的 pip 层、不再连累 torch 重下（~200MB+，慢）。
RUN pip install --default-timeout=600 --no-cache-dir \
    torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple

COPY requirements.txt .

# 2) 修复 requirements.txt 内在冲突:
#    - 注释 torch（CPU 已装）
#    - 注释 langchain-chroma==0.1.4（与 chromadb 1.5.5 不兼容；升级会牵连 langchain-core 1.x）
#      副作用：RAG mode 不可用，bot mode 不受影响
RUN sed -i 's/^torch==/#torch==/' requirements.txt && \
    sed -i 's/^langchain-chroma==/#langchain-chroma==/' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs sessions vector_db models/model_cache

EXPOSE 8088

CMD ["python", "main.py", "--mode", "bot"]
