#!/bin/bash
# =============================================================================
# pet-agent 服务器初始化脚本
# 在新服务器上运行一次即可
# =============================================================================

set -e

echo "=== pet-agent 服务器初始化 ==="

# 1. 安装 Docker
if ! command -v docker &> /dev/null; then
    echo "安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# 2. 安装 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "安装 Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

# 3. 创建部署目录
echo "创建部署目录..."
mkdir -p /opt/pet-agent
mkdir -p /opt/pet-agent/nginx/ssl

# 4. 配置防火墙 (如果使用)
if command -v firewall-cmd &> /dev/null; then
    echo "配置防火墙..."
    firewall-cmd --permanent --add-service=http
    firewall-cmd --permanent --add-service=https
    firewall-cmd --reload
elif command -v ufw &> /dev/null; then
    echo "配置 UFW..."
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw reload
fi

# 5. 创建 systemd 服务 (可选)
echo "创建 systemd 服务..."
cat > /etc/systemd/system/pet-agent.service << 'EOF'
[Unit]
Description=pet-agent Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/pet-agent
ExecStart=/usr/local/bin/docker-compose -f /opt/pet-agent/docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f /opt/pet-agent/docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
systemctl daemon-reload
systemctl enable pet-agent

echo ""
echo "=== 初始化完成 ==="
echo "下一步:"
echo "1. 将项目文件复制到 /opt/pet-agent"
echo "2. 配置 .env 文件 (不要提交到 Git)"
echo "3. 运行: cd /opt/pet-agent && docker-compose -f docker-compose.prod.yml up -d"
echo "4. 检查状态: curl http://localhost:8000/healthz"
