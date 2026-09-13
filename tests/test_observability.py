"""LangSmith 观测与 traceId 关联的测试。**全部离线**（不建连接、不发请求）。

## 为什么这组测试重要

观测是**非正确性路径** —— 它坏了对话仍然要能跑完。所以这里最需要钉住的
不是「能不能采数据」，而是两件事：

1. **不该采的时候一定不采。** `TracingConfig` 的缺省规则是「有 key 就启用」，
   而跑测的机器上可能恰好配了 key。所以「显式关必须压过 key」是一条安全断言，
   不是偏好。
2. **`trace_id` 与 `run_id` 同源。** 这是「traceId 和 LangSmith id 做关联」
   的全部实现方式 —— 一旦有人把 `run_id` 改成随机值，关联就静默断了，
   而本地一切照常。
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.observability.langsmith import (
    DEFAULT_PROJECT,
    ENV_TRACING,
    TracingConfig,
    build_tracer,
    resolve_run_id,
    run_config,
)

_KEY_ENVS = (
    "PET_AGENT_LANGSMITH_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name in _KEY_ENVS or name.startswith(
            ("PET_AGENT_", "LANGSMITH_", "LANGCHAIN_")
        ):
            monkeypatch.delenv(name, raising=False)


class TestTracingConfig:
    def test_disabled_without_key(self):
        assert TracingConfig.from_env().enabled is False

    def test_enabled_when_key_present(self, monkeypatch):
        monkeypatch.setenv("PET_AGENT_LANGSMITH_API_KEY", "ls__k")
        config = TracingConfig.from_env()
        assert config.enabled is True
        assert config.usable is True

    def test_explicit_off_beats_key(self, monkeypatch):
        """**CI 安全断言**：环境里恰好有 key 时，显式关必须赢。"""
        monkeypatch.setenv("PET_AGENT_LANGSMITH_API_KEY", "ls__k")
        monkeypatch.setenv(ENV_TRACING, "0")
        config = TracingConfig.from_env()
        assert config.enabled is False
        assert config.usable is False

    def test_explicit_on_without_key_is_not_usable(self, monkeypatch):
        """「开了但没 key」是配置错误 —— 不能算「已启用」。"""
        monkeypatch.setenv(ENV_TRACING, "1")
        config = TracingConfig.from_env()
        assert config.enabled is True
        assert config.usable is False

    def test_project_default_and_override(self, monkeypatch):
        assert TracingConfig.from_env().project == DEFAULT_PROJECT
        monkeypatch.setenv("LANGSMITH_PROJECT", "another-project")
        assert TracingConfig.from_env().project == "another-project"

    def test_describe_never_leaks_key(self, monkeypatch):
        monkeypatch.setenv("PET_AGENT_LANGSMITH_API_KEY", "ls__super-secret-value")
        described = str(TracingConfig.from_env().describe())
        assert "ls__super-secret-value" not in described
        assert "langsmith_usable" in described


class TestRunIdCorrelation:
    def test_uuid_trace_id_is_used_verbatim(self):
        """`trace_id` 默认就是 uuid4 → 二者是**同一个标识**，不是映射。"""
        trace_id = str(uuid.uuid4())
        assert resolve_run_id(trace_id) == uuid.UUID(trace_id)

    def test_non_uuid_is_derived_deterministically(self):
        """客户端自定义 id 也要能关联 —— 用确定性派生，而不是随机或映射表。"""
        first = resolve_run_id("client-side-id")
        second = resolve_run_id("client-side-id")
        assert first == second
        assert isinstance(first, uuid.UUID)

    def test_different_ids_do_not_collide(self):
        assert resolve_run_id("a") != resolve_run_id("b")

    def test_run_config_carries_the_same_id(self):
        trace_id = str(uuid.uuid4())
        config = run_config(
            trace_id=trace_id, session_id="s-1", user_id="u-1", pet_id="p-1"
        )
        assert config["run_id"] == uuid.UUID(trace_id)
        assert config["metadata"]["trace_id"] == trace_id
        assert config["metadata"]["session_id"] == "s-1"

    def test_run_config_without_tracer_has_no_callbacks(self):
        config = run_config(trace_id="t", session_id="s", user_id="u", pet_id="p")
        assert "callbacks" not in config


class TestBuildTracer:
    def test_disabled_returns_none_without_error(self):
        tracer, error = build_tracer(TracingConfig(enabled=False))
        assert tracer is None
        assert error is None

    def test_enabled_without_key_reports_the_reason(self):
        """开了没 key 必须**说出原因** —— 静默不采比报错更难发现。"""
        tracer, error = build_tracer(TracingConfig(enabled=True, api_key=None))
        assert tracer is None
        assert error is not None
        assert "key" in error.lower()

    def test_with_key_builds_tracer(self):
        tracer, error = build_tracer(
            TracingConfig(enabled=True, api_key="ls__fake", project="pet-agent")
        )
        assert error is None
        assert tracer is not None
