FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（chromadb 需要 sqlite3，sentence-transformers 需要 gcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存（代码改动不重装包）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目代码
COPY . .

# 创建运行时目录
RUN mkdir -p logs sessions vector_db models/model_cache

EXPOSE 8088

CMD ["python", "main.py", "--mode", "bot"]