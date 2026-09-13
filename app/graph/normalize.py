"""输入归一化。

**本模块解决了 docs/10-self-review.md 的 U2/F6**（``understand_input`` 的时间归一化职责未定义）。

为什么必须有它：``DESIGN.md`` §3.5 把「时间范围重叠」定为**冲突判定的必要条
件**。如果「上个月」没有被解析成具体日期，冲突判定就会把「上个月喜欢」
与「这周不喜欢」误判为矛盾 —— 而它们其实是**演变**。

⚠️ **当前是规则实现，覆盖有限**。只处理相对时间的常见表达。
精确解析需要更完整的时序解析器（或在生产环境用模型抽取 + 代码校验）。
未识别的表达一律返回 ``None``（无界）—— **不猜**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TimeRange:
    """一个时间区间。``None`` 表示该侧无界。"""

    start: datetime | None = None
    end: datetime | None = None

    @property
    def is_bounded(self) -> bool:
        return self.start is not None or self.end is not None


#: 相对时间表达 → (起始偏移天数, 结束偏移天数)。偏移以「今天」为 0，负数表示过去。
#: ``None`` 表示该侧无界（例：「一直」的起始侧）。
#: 覆盖常见的家庭对话表达；未列出的表达一律不解析。
_RELATIVE_PATTERNS: tuple[tuple[str, tuple[int | None, int | None]], ...] = (
    ("今天", (0, 0)),
    ("昨天", (-1, -1)),
    ("前天", (-2, -2)),
    ("这周", (-6, 0)),
    ("本周", (-6, 0)),
    ("上周", (-13, -7)),
    ("这个月", (-29, 0)),
    ("本月", (-29, 0)),
    ("上个月", (-59, -30)),
    ("上月", (-59, -30)),
    ("最近", (-7, 0)),
    ("这几天", (-6, 0)),
    ("这些天", (-6, 0)),
    ("前几天", (-10, -3)),
    ("一直", (None, 0)),  # 特例：从很早到现在的持续状态
)

#: 显式日期：2026-09-01 / 2026/9/1
_ISO_DATE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")


def parse_time_range(text: str | None, *, now: datetime | None = None) -> TimeRange | None:
    """从文本解析时间范围。

    Returns:
        ``TimeRange``；无法识别时返回 ``None``。
        **不猜**：识别不了就是 ``None``（调用方按「无界 = 当前有效」处理）。
    """
    if not text:
        return None

    ref = now or _now()

    match = _ISO_DATE.search(text)
    if match:
        try:
            start = datetime(
                int(match.group(1)), int(match.group(2)), int(match.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return None
        start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return TimeRange(start=start_day, end=start_day + timedelta(days=1))

    for keyword, (start_offset, end_offset) in _RELATIVE_PATTERNS:
        if keyword not in text:
            continue
        base = ref.replace(hour=0, minute=0, second=0, microsecond=0)
        start = None if start_offset is None else base + timedelta(days=start_offset)
        # 结束侧取当天 23:59:59，使「今天」覆盖整天
        end = (
            base + timedelta(days=end_offset) + timedelta(days=1) - timedelta(seconds=1)
            if end_offset is not None
            else None
        )
        return TimeRange(start=start, end=end)

    return None
