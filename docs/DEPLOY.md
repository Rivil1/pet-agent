# pet-agent 部署指南

## 目录

- [架构](#架构)
- [快速部署](#快速部署)
- [环境变量](#环境变量)
- [CI/CD](#cicd)
- [本项目的两个环境约束](#本项目的两个环境约束)
- [运维](#运维)
- [故障排查](#故障排查)

---

## 架构

```
                 ┌─────────────────────────────────────────┐
   公网 :80 ───► │  pet-web  (nginx + 前端静态产物)          │
                 │   ├─ /            静态文件（Vite 构建产物）│
                 │   ├─ /v1/*   ─┐                          │
                 │   └─ /healthz ─┴─► 反代                   │
                 └──────────────────┬──────────────────────┘
                                    │ 容器网络 pet-net
                                    ▼
                 ┌─────────────────────────────────────────┐
                 │  pet-agent  (uvicorn, 只绑 127.0.0.1)     │
                 │   LLM/视觉 → 火山方舟 ARK                  │
                 │   向量      → 百炼 text-embedding-v4      │
                 └─────────────────────────────────────────┘

   # 下面三个默认不启动（compose profile = data）
   # 存储层目前仍是 InMemoryStore，起了也接不上
   mysql (主库) · milvus (向量) · redis (缓存)
```

**为什么后端不对外暴露端口**：所有流量都应经过同一层。直连 8000 的路径
没有安全头、没有压缩、没有统一超时 —— 那是一个「只在生产存在」的差异，
而这类差异最容易在生产被利用。

**为什么前端与 Nginx 同镜像**：前端是纯静态产物，不需要独立运行时；
同源还顺带消掉了 CORS 配置和「把后端地址编译进产物」的需求
（换环境不用重新构建）。

---

## 快速部署

### 一键部署（本地 → 服务器）

```bash
# 默认目标写在脚本里；可用环境变量覆盖
SERVER=47.102.186.248 SSH_KEY=~/aaa.pem scripts/deploy.sh

# 只同步代码不重建镜像
scripts/deploy.sh --no-build
```

脚本会：本地构建预检 → 打包 → 上传 → 备份 `.env` → 同步 → 服务器本地构建 → 重启 → 健康检查。

### 首次在新服务器上部署

```bash
# 1. 装 Docker
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# 2. 配 Docker 镜像源（境内必需）
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://docker.m.daocloud.io", "https://public.ecr.aws"]
}
EOF
systemctl restart docker

# 3. 建目录并放入 .env（见下方「环境变量」）
mkdir -p /opt/pet-agent && cd /opt/pet-agent

# 4. 从本地推一次代码
scripts/deploy.sh
```

---

## 环境变量

`.env` **只存在于服务器**，且在 `.gitignore` 里 —— 任何代码同步路径都拿不到它，
所以部署脚本会先备份它。丢了就只能重新申请密钥。

### 运行时必需

| 变量 | 说明 |
|---|---|
| `PET_AGENT_AUTH_SECRET` | 鉴权签名密钥。**缺失则拒绝启动** —— 没有它 `user_id` 可伪造，租户隔离不成立 |
| `ARK_API_KEY` | 火山方舟 Key（LLM + 视觉） |
| `PET_AGENT_EMBED_API_KEY` | 百炼 Key（向量） |

### 构建源（境内服务器必需）

| 变量 | 值 |
|---|---|
| `DEBIAN_MIRROR` | `mirrors.aliyun.com` |
| `ALPINE_MIRROR` | `mirrors.aliyun.com` |
| `PIP_INDEX_URL` | `https://mirrors.aliyun.com/pypi/simple/` |
| `NPM_REGISTRY` | `https://registry.npmmirror.com` |

> 不配这些，构建会卡在 `deb.debian.org`（实测：拉了 5 分钟没动）。
> 云端 CI 不传这些参数即可走官方源 —— Dockerfile 里都是 `ARG`。

### ⚠️ 生产环境必须删掉的变量

| 变量 | 为什么 |
|---|---|
| `PET_AGENT_ALLOW_DEV_LOGIN=1` | 它允许为**任意 `user_id`** 换取合法 token。开着它就等于取消 D5（多租户隔离）。仅用于演示环境 |

生成 `PET_AGENT_AUTH_SECRET`：

```bash
openssl rand -hex 32
```

---

## CI/CD

`.github/workflows/ci.yml`：

```
backend-lint ─┐
backend-test ─┼─► frontend-e2e ─► build-images ─► deploy
frontend-lint ┘
```

| Job | 内容 |
|---|---|
| `backend-lint` | ruff + pyright |
| `backend-test` | pytest（640 条） |
| `frontend-lint` | tsc --noEmit + vite build |
| `frontend-e2e` | 起真后端（**mock provider，离线可跑**）+ 真构建产物，跑 22 项断言 |
| `build-images` | 构建并推送后端/前端镜像到 GHCR |
| `deploy` | 打包 → SCP 到服务器 → 服务器本地构建 → 重启 → 健康检查 |

### 需要的配置

**Secrets**（Settings → Secrets and variables → Actions → Secrets）

| 名称 | 说明 |
|---|---|
| `SERVER_SSH_KEY` | 部署用的 SSH 私钥 |
| `PET_AGENT_AUTH_SECRET` | 可选，若想在 CI 里校验 |
| `ARK_API_KEY` / `PET_AGENT_EMBED_API_KEY` | 可选，CI 用 mock 时不需要 |

**Variables**

| 名称 | 示例 |
|---|---|
| `SERVER_HOST` | `47.102.186.248` |
| `SERVER_USER` | `root` |
| `SERVER_PORT` | `22` |

---

## 本项目的两个环境约束

这两条都是实测出来的，不是假设。不知道它们会浪费很多时间。

### 1. 服务器连不上 GitHub

```
$ git fetch origin main
fatal: unable to access 'https://github.com/Rivil1/pet-agent.git/':
       Empty reply from server
```

所以 CI 里**不能**用「服务器 `git pull`」的部署方式。
`deploy` job 改为：本地打包 → `scp` 推过去 → 服务器解包。

注意 GitHub Actions 本身能正常访问 GitHub，所以整条链路是通的 ——
只有「服务器主动拉」这一段不行。

### 2. 基础镜像拉取很慢

`python:3.11-slim` 约 30MB 要 2–3 分钟，`nginx:alpine` 52MB 要 5 分钟以上。
国内 registry 镜像能缓解但仍有节流。

所以**不要在每次部署时重建镜像**。用 `scripts/deploy.sh --no-build`
在只改代码不改编排配置时跳过。

---

## 运维

```bash
cd /opt/pet-agent

# 状态
docker compose -f docker-compose.prod.yml ps

# 日志
docker compose -f docker-compose.prod.yml logs -f pet-agent
docker compose -f docker-compose.prod.yml logs -f web

# 重启（不重建）
docker compose -f docker-compose.prod.yml restart pet-agent

# 完全重建
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d --build

# 进入后端容器排查
docker compose -f docker-compose.prod.yml exec pet-agent sh

# 健康检查
curl -s http://127.0.0.1/healthz | python3 -m json.tool

# 签发一个 token（生产路径，不需要开 dev-login）
docker compose -f docker-compose.prod.yml exec pet-agent \
  python -m app.bootstrap --issue-token <user-id>

# 查看当前装配到的厂商 / 模型 / 端点
docker compose -f docker-compose.prod.yml exec pet-agent \
  python -m app.bootstrap --describe
```

### 启用数据服务（MySQL / Milvus / Redis）

```bash
# 先在 .env 里设好 MYSQL_PASSWORD / MYSQL_ROOT_PASSWORD
docker compose -f docker-compose.prod.yml --profile data up -d
```

> ⚠️ 目前存储层仍是 `InMemoryStore`（`app/store/memory.py`），
> 起了这些容器也**不会**被使用。接入前它们只是占资源。

---

## 故障排查

### 部署后页面白屏

```bash
# 1. 看容器是否在跑
docker compose -f docker-compose.prod.yml ps

# 2. 看 nginx 能不能拿到静态文件
docker compose -f docker-compose.prod.yml exec web ls /usr/share/nginx/html

# 3. 看 nginx 配置有没有语法错
docker compose -f docker-compose.prod.yml exec web nginx -t
```

### 接口 502

后端没起来，或健康检查没过：

```bash
docker compose -f docker-compose.prod.yml logs --tail=60 pet-agent
# 常见原因：PET_AGENT_AUTH_SECRET 缺失（会显式拒绝启动）
```

### 构建卡住

看卡在哪个源：

```bash
tail -20 /root/build.log
```

- 卡 `deb.debian.org` → 没设 `DEBIAN_MIRROR`
- 卡 `registry.npmjs.org` → 没设 `NPM_REGISTRY`
- 卡 `pypi.org` → 没设 `PIP_INDEX_URL`
- 卡在拉基础镜像 → 正常，换个时间或等

### 模型输出被当成真实结论

`/healthz` 的 `providers.mode`：

- `live` — 真模型
- `mock` — 占位实现，输出是写死的
- `unknown` — 无法确认（**与 mock 同等对待**）

前端顶栏常驻这个状态，就是为了让「假演示」不能静默发生。
