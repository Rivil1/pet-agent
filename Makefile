# =============================================================================
# pet-agent
#
# `docs/DESIGN.md` §6.5 要求一键可复现：
#
#     make demo       # 起服务 + 灌样例数据 + 打开 Web UI
#     make eval       # 跑全部评测，输出报告
#     make ablation   # 跑消融对比，输出表格
#
# 这个 Makefile 就是那三条的实现。
#
# ## 一条贯穿的设计约束
#
# **评测不需要 API Key、不需要网络。** 它用 `HashEmbedder` + `MockLLM`，
# 所以任何人 clone 下来就能跑出同样的数字（`DESIGN §6.5`）。
# 需要真模型的部分会被显式标成「不可测」，而不是假装测过。
# =============================================================================

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
PORT   ?= 8000
OUT    ?= reports

.DEFAULT_GOAL := help
.PHONY: help install test test-all lint typecheck eval ablation eval-list \
        serve demo tunnel deploy clean fmt report

help:  ## 显示可用目标
	@echo "pet-agent"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── 安装与检查 ───────────────────────────────────────────────

install:  ## 安装依赖
	$(PIP) install -r requirements.txt

lint:  ## 静态检查（ruff）
	$(PYTHON) -m ruff check app/ tests/ || true

typecheck:  ## 类型检查（pyright）
	$(PYTHON) -m pyright app/

fmt:  ## 格式化
	$(PYTHON) -m ruff format app/ tests/ || true

# ── 测试 ─────────────────────────────────────────────────────

test:  ## 跑测试（不需要数据库/网络）
	$(PYTHON) -m pytest tests/ -q

test-all:  ## 跑测试，含 MySQL 集成组（需先 make tunnel 或配好 MYSQL_*）
	$(PYTHON) -m pytest tests/ -q

# ── 评测（本项目的核心交付） ─────────────────────────────────

eval:  ## 跑评测基线，输出 reports/eval.md
	$(PYTHON) -m app.eval --out $(OUT)

eval-list:  ## 列出全部评测场景与消融配置
	$(PYTHON) -m app.eval --list

ablation:  ## 跑基线与全部消融组，输出对比
	$(PYTHON) -m app.eval --ablation --out $(OUT)

eval-strict:  ## 跑评测并在有断言失败时以非零码退出（CI 用）
	$(PYTHON) -m app.eval --strict --out $(OUT)

report: eval  ## eval 的别名
	@echo ""
	@echo "报告：$(OUT)/eval.md"

# ── 运行 ─────────────────────────────────────────────────────

serve:  ## 起后端（开发登录开启，便于本地联调）
	PET_AGENT_ALLOW_DEV_LOGIN=1 \
	$(PYTHON) -m uvicorn app.bootstrap:create_app_from_env \
	  --factory --host 0.0.0.0 --port $(PORT)

demo:  ## 一键演示：起后端 + 打印可用的入口与 token
	@echo "启动后端（Ctrl-C 停止）…"
	@echo ""
	@PET_AGENT_ALLOW_DEV_LOGIN=1 $(PYTHON) -m uvicorn app.bootstrap:create_app_from_env \
	  --factory --host 0.0.0.0 --port $(PORT) & \
	 sleep 4; \
	 echo "  API 文档   http://localhost:$(PORT)/docs"; \
	 echo "  健康检查   http://localhost:$(PORT)/healthz"; \
	 echo ""; \
	 echo "  开发 token（所有端点都需要 Bearer token）："; \
	 PET_AGENT_ALLOW_DEV_LOGIN=1 $(PYTHON) -m app.bootstrap --issue-token demo-user 2>/dev/null | sed 's/^/    /' || true; \
	 wait

# ── 运维 ─────────────────────────────────────────────────────

tunnel:  ## 本地连线上数据库（SSH 隧道，前台）
	bash scripts/db-tunnel.sh

tunnel-up:  ## 同上，后台
	bash scripts/db-tunnel.sh --daemon

tunnel-down:  ## 停掉隧道
	bash scripts/db-tunnel.sh --stop

deploy:  ## 部署到服务器
	bash scripts/deploy.sh

clean:  ## 清理缓存与临时产物
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	rm -rf $(OUT)
