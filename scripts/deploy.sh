#!/bin/bash
# =============================================================================
# pet-agent 部署脚本
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is required but not installed."
        exit 1
    fi
}

# 主部署函数
deploy() {
    log_info "Starting deployment..."
    
    # 检查依赖
    check_command docker
    check_command docker-compose
    
    # 获取脚本目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    cd "$PROJECT_DIR"
    
    # 拉取最新代码
    if [ -d ".git" ]; then
        log_info "Pulling latest code..."
        git pull origin main
    fi
    
    # 构建 Docker 镜像
    log_info "Building Docker image..."
    docker build -t pet-agent:latest .
    
    # 停止旧容器
    log_info "Stopping old containers..."
    docker-compose -f docker-compose.prod.yml down || true
    
    # 启动新服务
    log_info "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # 等待服务启动
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # 检查健康状态
    log_info "Checking health status..."
    if curl -f http://localhost:8000/healthz &> /dev/null; then
        log_info "✅ Service is healthy!"
    else
        log_error "❌ Service health check failed!"
        docker-compose -f docker-compose.prod.yml logs pet-agent
        exit 1
    fi
    
    # 清理旧镜像
    log_info "Cleaning up old images..."
    docker image prune -f
    
    log_info "🎉 Deployment completed successfully!"
}

# 回滚函数
rollback() {
    log_warn "Rolling back to previous version..."
    
    docker-compose -f docker-compose.prod.yml down
    docker image prune -f
    
    log_info "Rollback completed. Previous version is still running if exists."
}

# 显示状态
status() {
    echo "=== Container Status ==="
    docker-compose -f docker-compose.prod.yml ps
    
    echo ""
    echo "=== Health Check ==="
    curl -s http://localhost:8000/healthz | python3 -m json.tool 2>/dev/null || echo "Service not responding"
    
    echo ""
    echo "=== Recent Logs ==="
    docker-compose -f docker-compose.prod.yml logs --tail=20 pet-agent
}

# 显示使用帮助
usage() {
    echo "Usage: $0 {deploy|rollback|status|logs}"
    echo ""
    echo "Commands:"
    echo "  deploy    - Deploy the application"
    echo "  rollback  - Rollback to previous version"
    echo "  status    - Show service status"
    echo "  logs      - Show application logs"
}

# 主入口
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    status)
        status
        ;;
    logs)
        docker-compose -f docker-compose.prod.yml logs -f pet-agent
        ;;
    *)
        usage
        exit 1
        ;;
esac
