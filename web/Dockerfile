# =============================================================================
# pet-agent 前端：构建 → 静态服务 + API 反代
#
# 为什么把前端与 Nginx 打进同一个镜像：
#   前端是**纯静态产物**，它不需要独立的运行时进程。
#   分成两个容器会多一层网络跳转，而收益只是「前端可以单独扩容」——
#   这个应用不需要。同镜像同源还顺带消掉了 CORS 问题。
# =============================================================================

# ---------- 构建阶段 ----------
FROM node:22-alpine AS builder

# npm 源可替换：云端 CI 用官方源，国内服务器用镜像 ——
# 写死任何一个都会在另一半环境里变成奇慢或直接失败。
ARG NPM_REGISTRY=https://registry.npmjs.org
# Alpine 的 apk 源同理
ARG ALPINE_MIRROR=dl-cdn.alpinelinux.org
RUN set -eux; \
    if [ "$ALPINE_MIRROR" != "dl-cdn.alpinelinux.org" ]; then \
      sed -i "s|dl-cdn.alpinelinux.org|$ALPINE_MIRROR|g" /etc/apk/repositories; \
    fi; \
    npm config set registry "$NPM_REGISTRY" --global

WORKDIR /build

# 先只拷依赖清单：源码变动时这一层仍能命中缓存
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY . .

# 构建期不注入后端地址 —— 生产是同源部署，走相对路径最简单也最不易错
RUN npm run build

# ---------- 运行阶段 ----------
FROM nginx:1.27-alpine

# 移除默认站点，换成我们自己的（含 API 反代）
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/pet-agent.conf

# 静态产物
COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

# 健康检查必须写 127.0.0.1 而不是 localhost。
#
# 容器内 `localhost` 会先解析到 IPv6 `::1`，而 nginx 默认只监听 IPv4 ——
# 于是 `wget http://localhost/health` 直接 Connection refused，
# 健康状态连续失败，而站点实际上完全是好的。
# 这种「监控说挂了但服务是好的」比真挂更浪费排查时间。
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/health >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
