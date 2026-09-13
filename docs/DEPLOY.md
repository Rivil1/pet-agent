# pet-agent 部署指南

## 目录

- [快速部署](#快速部署)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [手动部署](#手动部署)
- [环境变量配置](#环境变量配置)
- [故障排查](#故障排查)

---

## 快速部署

### 1. 服务器初始化

```bash
# 在服务器上运行一次初始化脚本
curl -fsSL https://raw.githubusercontent.com/YOUR_USER/pet-agent/main/scripts/init-server.sh | bash
```

### 2. 配置环境变量

```bash
cd /opt/pet-agent
cp .env.prod .env
nano .env  # 编辑填入真实密钥
```

### 3. 启动服务

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## GitHub Actions CI/CD

### 设置步骤

#### 1. 配置 GitHub Secrets

在 GitHub 仓库 Settings → Secrets and variables → Actions 中添加:

| Secret 名称 | 说明 |
|-------------|------|
| `SERVER_SSH_KEY` | 服务器 SSH 私钥 |
| `ARK_API_KEY` | 火山方舟 API Key |
| `PET_AGENT_EMBED_API_KEY` | 百炼 Embedding API Key |
| `PET_AGENT_AUTH_SECRET` | 鉴权密钥 (随机字符串) |

#### 2. 配置 GitHub Variables

在 GitHub 仓库 Settings → Secrets and variables → Actions → Variables 中添加:

| Variable 名称 | 说明 | 示例值 |
|---------------|------|--------|
| `SERVER_HOST` | 服务器 IP | `47.102.186.248` |
| `SERVER_USER` | SSH 用户名 | `root` |
| `SERVER_PORT` | SSH 端口 | `22` |
| `PET_AGENT_EMBED_BASE_URL` | Embedding API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `PET_AGENT_EMBED_MODEL` | Embedding 模型名 | `text-embedding-v4` |

#### 3. 启用 Actions

推送代码到 main 分支将自动触发 CI/CD 流程:

```bash
git add .
git commit -m "feat: 添加 CI/CD 配置"
git push origin main
```

### CI/CD 流程

```
push → [Lint & Type Check] → [Unit Tests] → [Build Docker] → [Deploy to Server]
                                                           ↓
                                                    [Notify]
```

---

## 手动部署

### 服务器上手动部署

```bash
# 1. 安装依赖
yum install -y docker docker-compose

# 2. 启动 Docker
systemctl start docker
systemctl enable docker

# 3. 创建目录
mkdir -p /opt/pet-agent
cd /opt/pet-agent

# 4. 下载项目
git clone https://github.com/YOUR_USER/pet-agent.git .
git checkout main

# 5. 配置环境变量
cp .env.prod .env
nano .env  # 填入真实密钥

# 6. 启动服务
docker-compose -f docker-compose.prod.yml up -d

# 7. 检查状态
curl http://localhost:8000/healthz
docker-compose -f docker-compose.prod.yml ps
```

### 使用部署脚本

```bash
# 部署
./scripts/deploy.sh deploy

# 查看状态
./scripts/deploy.sh status

# 查看日志
./scripts/deploy.sh logs

# 回滚
./scripts/deploy.sh rollback
```

---

## 环境变量配置

### 必需变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `PET_AGENT_AUTH_SECRET` | 鉴权签名密钥 | `your-random-secret-here` |
| `ARK_API_KEY` | 火山方舟 API Key | `ark-xxxxxx` |
| `PET_AGENT_EMBED_API_KEY` | 百炼 API Key | `sk-xxxxxx` |

### 可选变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PET_AGENT_PROVIDER` | `ark` | LLM 供应商 |
| `PET_AGENT_EMBED_BASE_URL` | 百炼地址 | Embedding API 地址 |
| `PET_AGENT_EMBED_MODEL` | `text-embedding-v4` | Embedding 模型名 |
| `PET_AGENT_EMBED_DIM` | `1024` | Embedding 向量维度 |

---

## 故障排查

### 查看日志

```bash
# 应用日志
docker-compose -f docker-compose.prod.yml logs -f pet-agent

# Nginx 日志
docker exec pet-nginx tail -f /var/log/nginx/access.log

# 系统日志
journalctl -u pet-agent -f
```

### 重启服务

```bash
docker-compose -f docker-compose.prod.yml restart
```

### 完全重建

```bash
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d --build
```

### 健康检查

```bash
curl http://localhost:8000/healthz | python3 -m json.tool
```

---

## 生产环境注意事项

1. **安全**: 不要将 `.env` 文件提交到 Git
2. **SSL**: 取消注释 nginx.conf 中的 HTTPS 配置
3. **备份**: 定期备份 Redis 数据和宠物数据
4. **监控**: 建议配置 Grafana + Prometheus 监控
5. **日志**: 配置日志轮转避免磁盘占满
