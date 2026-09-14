# =============================================================================
# pet-agent 生产环境 Dockerfile
# =============================================================================

# 多阶段构建：构建阶段 + 运行阶段
FROM python:3.11-slim AS builder

# Debian 源可替换：deb.debian.org 在国内会被节流到几乎不可用，
# 而云端 CI 用官方源最快。硬编码哪个都会在另一半环境里退回慢速或失败。
ARG DEBIAN_MIRROR=deb.debian.org
RUN set -eux; \
    if [ "$DEBIAN_MIRROR" != "deb.debian.org" ]; then \
      sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
      sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list; \
    fi

WORKDIR /build

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .

# 源可替换（同 web/Dockerfile）：云端 CI 与国内服务器需要不同的索引
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --user --index-url "$PIP_INDEX_URL" -r requirements.txt

# 运行阶段
FROM python:3.11-slim

# 运行阶段同样需要换源（详见构建阶段的说明）
ARG DEBIAN_MIRROR=deb.debian.org
RUN set -eux; \
    if [ "$DEBIAN_MIRROR" != "deb.debian.org" ]; then \
      sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
      sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g" /etc/apt/sources.list; \
    fi

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# 复制 Python 包
COPY --from=builder /root/.local /home/appuser/.local

# 复制应用代码
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser pyrightconfig.json .

# 上传媒体的落盘目录。
#
# ⚠️ **必须在镜像里建好并 chown 给 appuser。**
# 容器以非 root 的 `appuser` 运行，而 `/app` 归 root ——
# 让应用在启动时 `mkdir /app/media` 会 PermissionError，
# 而那个报错发生在**导入期**，看起来像代码坏了。
RUN mkdir -p /app/media && chown appuser:appuser /app/media

# 环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH
ENV USER=appuser

# 切换到非 root 用户
USER appuser

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# 入口必须是运行时装配层（见 app/bootstrap.py）
CMD ["uvicorn", "app.bootstrap:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
