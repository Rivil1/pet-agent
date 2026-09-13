"""多模态模型的事实提取。

**模型做观察，代码做测量。** 两者的分界不是「谁说的」，而是**「能不能重算」**：

| | 产出 | 可复现 | 契约 |
| --- | --- | --- | --- |
| `app/audio/features.py` | `AcousticFeatures` | ✅ | `MEASURED` |
| `app/extract/multimodal.py` | `ModelObservation` | ❌ | `OBSERVED` |

契约层强制这一点：`MEASURED` 必须带 `value`（见 `app/schemas/behavior.py`）。

## 本期策略

**先用多模态模型跑通，后续在真实调用中优化。**

原本设想的自定义管线（解封装 / 人声过滤 / 时间戳对齐 / 抽帧）
是在**没有任何真实数据**的情况下设计的，其中的人声判据尤其容易写宽或写窄 ——
而那些阈值只能靠真实样本调。先让模型跑，看真实的输入输出分布，
再决定哪一段值得写成代码。

## 铁律：模型不得用来做这三件事

| 不做 | 理由 |
| --- | --- |
| **推断因果**（「开门让它停了」） | 时间相邻 ≠ 因果，且 `resolution` 必须由主人提供 |
| **判断神态**（`general.demeanor`） | 它是**红旗信号**，用未校验的推断驱动它，漏报代价是猫可能死亡 |
| **伪装成测量** | 契约会拒绝没有 `value` 的 `MEASURED` 项 |
"""

from app.extract.multimodal import (
    MultimodalExtractor,
    dump_observation,
    parse_model_observation,
)

__all__ = [
    "MultimodalExtractor",
    "parse_model_observation",
    "dump_observation",
]
