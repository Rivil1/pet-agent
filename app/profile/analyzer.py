"""档案分析：多图取交集 → 稳定身份特征。

对应 docs/DESIGN.md §1.3「身份锚定」与 §2.3 ``profile_analyzer`` 节点契约。

**这个模块实现的是整个项目最核心的身份机制**：
从多张照片中区分「一致出现的特征」（进 ``must_keep_features``，作为硬约束）
与「只出现一次的特征」（进 ``observed_but_unstable``，仅供参考）。

为什么这一步必须由**代码**做、不能交给模型：
1. 交集运算是确定性的，模型做会有随机性 —— 同一批照片两次可能给出不同档案
2. 档案是**长期身份锚点**，它的稳定性直接决定「不同对话中 AI 认识的是同一只猫」
3. 这一步同时是「外观一致性校验」的基准（`DESIGN.md` §1.5 边界 1）

⚠️ 注意边界：``must_keep_features`` 是**外观一致性的文本约束**，
**不是个体身份鉴定**。通用视觉模型的 embedding 做不到个体再识别。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.schemas import PetProfileDraft, VisualProfile


@dataclass(frozen=True)
class VisualObservation:
    """单张照片的视觉观察结果。

    ``ok=False`` 表示这张照片分析失败 —— **它不参与交集运算**，
    但会被记入 ``failed_photo_urls`` 并体现在覆盖度里。
    """

    image_url: str
    fur_color: str | None = None
    fur_length: str | None = None
    eye_color: str | None = None
    body_shape: str | None = None
    face_shape: str | None = None
    distinctive_features: tuple[str, ...] = ()
    ok: bool = True
    error: str | None = None

    def terms(self) -> list[str]:
        """展平为词项列表，只含非空项。"""
        out: list[str] = []
        for value in (self.fur_color, self.fur_length, self.eye_color, self.body_shape, self.face_shape):
            if value:
                out.append(value.strip())
        out.extend(f.strip() for f in self.distinctive_features if f and f.strip())
        return out


@dataclass(frozen=True)
class FeatureConsensus:
    """交集结果。"""

    stable: list[str]
    """在 ≥ ``min_support`` 张照片中一致出现 → 进 ``must_keep_features``。"""

    unstable: list[str]
    """出现次数不足 → 仅供参考，**不作为身份硬约束**。"""

    counts: dict[str, int] = field(default_factory=dict)

    def explain(self) -> str:
        if not self.counts:
            return "没有任何可用特征"
        stable_txt = "、".join(self.stable) if self.stable else "无"
        unstable_txt = "、".join(self.unstable) if self.unstable else "无"
        return f"稳定特征：{stable_txt}；仅部分照片出现：{unstable_txt}"


@runtime_checkable
class VisionAnalyzer(Protocol):
    """单张照片的视觉分析。"""

    def analyze(self, image_url: str) -> VisualObservation: ...


def intersect_observations(
    observations: list[VisualObservation], *, min_support: int = 2
) -> FeatureConsensus:
    """多图取交集。

    Args:
        min_support: 判为「稳定」所需的最少出现次数。

    **只统计成功的观察**。失败的照片既不算支持、也不算反对 ——
    把它们计为「未出现」会系统性地降低所有特征的支持度。
    """
    successful = [o for o in observations if o.ok]
    counter: Counter[str] = Counter()
    for obs in successful:
        # 同一张照片里的重复词项只算一次
        counter.update(set(obs.terms()))

    stable = sorted(t for t, n in counter.items() if n >= min_support)
    unstable = sorted(t for t, n in counter.items() if n < min_support)
    return FeatureConsensus(stable=stable, unstable=unstable, counts=dict(counter))


def build_draft(
    observations: list[VisualObservation],
    *,
    name: str | None = None,
    breed: str | None = None,
    min_support: int = 2,
) -> PetProfileDraft:
    """构建待用户确认的档案草案。

    覆盖度与失败照片必须如实呈现 ——
    如果只基于 2/5 张照片就建出档案却不说明，用户会以为系统看全了。
    """
    successful = [o for o in observations if o.ok]
    failed = [o.image_url for o in observations if not o.ok]

    if not successful:
        # 全部失败 → 明确报错，**不编造特征**（DESIGN.md §2.3 失败行为）
        raise ValueError(
            f"全部 {len(observations)} 张照片分析失败，无法建立档案。"
            "请重新上传清晰、正面、光线充足的照片。"
        )

    consensus = intersect_observations(observations, min_support=min_support)

    visual = VisualProfile(
        **{
            field_name: _most_common(getattr(o, field_name) for o in successful)
            for field_name in (
                "fur_color",
                "fur_length",
                "eye_color",
                "body_shape",
                "face_shape",
            )
        },
        distinctive_features=consensus.stable,
    )

    coverage_note = None
    if failed:
        coverage_note = (
            f"基于 {len(successful)}/{len(observations)} 张照片"
            f"（{len(failed)} 张分析失败，未参与特征提取）"
        )

    return PetProfileDraft(
        name=name,
        breed=breed,
        visual=visual,
        must_keep_features=consensus.stable,
        observed_but_unstable=consensus.unstable,
        photo_count=len(observations),
        analyzed_photo_count=len(successful),
        failed_photo_urls=failed,
        coverage_note=coverage_note,
    )


def _most_common(values) -> str | None:
    """取众数。并列时取字典序最小者 —— **保证确定性**（同一输入 → 同一档案）。"""
    cleaned = [v.strip() for v in values if v and v.strip()]
    if not cleaned:
        return None
    counter = Counter(cleaned)
    top = max(counter.values())
    return sorted(t for t, n in counter.items() if n == top)[0]


def identify_from_photos(
    image_urls: list[str],
    analyzer: VisionAnalyzer,
    *,
    name: str | None = None,
    breed: str | None = None,
    min_support: int = 2,
) -> PetProfileDraft:
    """端到端：分析多张照片 → 档案草案。

    单张失败**不中断整体**（跳过并记录），全部失败才报错。
    """
    observations: list[VisualObservation] = []
    for url in image_urls:
        try:
            observations.append(analyzer.analyze(url))
        except Exception as exc:  # noqa: BLE001 — 单张失败必须被隔离
            observations.append(
                VisualObservation(image_url=url, ok=False, error=f"{type(exc).__name__}: {exc}")
            )
    return build_draft(observations, name=name, breed=breed, min_support=min_support)
