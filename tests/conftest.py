"""pytest 配置：把项目根加入 import 路径，使 `import app.schemas` 可用。

用法：
    /root/.venvs/pet-agent/bin/python -m pytest        # 在项目根执行
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 观测（LangSmith）在测试中**强制关闭**：
#
# 1. 测试必须离线、可复现；
# 2. `TracingConfig.from_env` 的缺省规则是「有 key 就启用」——
#    若跑测的机器上恰好配了 LangSmith key，测试就会真的外发数据。
#    那条路径不会让断言失败，所以是静默的。
# 用**赋值而不是 setdefault**：显式关闭必须能压过环境变量。
os.environ["PET_AGENT_TRACING"] = "0"
