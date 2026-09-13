"""宠物档案契约。

核心设计：**三层信息分离**（见 docs/00-product.md §3.3）。
观察事实 / AI 推测 / 拟人化设定 在数据模型层面就是分开的，不靠 prompt 约束。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Species(str, Enum):
    CAT = "cat"


class TraitLayer(str, Enum):
    """三层信息分离的枚举化。混用这三层是产品可信度的最大威胁。"""

    OBSERVED = "observed"
    """用户直接陈述或系统直接测量的事实。例：'它每天晚上十点左右跑酷'。"""

    INFERRED = "inferred"
    """模型基于证据的推断。例：'它可能在寻求互动'。"""

    ROLEPLAY = "roleplay"
    """用户为该猫设定的拟人化风格。例：'说话傲娇'。不承载事实。"""


class Trait(BaseModel):
    """一条关于这只猫的描述，带明确的层级归属与来源。"""

    layer: TraitLayer
    text: str
    source: str = Field(
        description="来源标识，如 user_statement / model_inference / user_setting / system_measurement"
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="仅 INFERRED 层需要。OBSERVED 层应为 None（事实不需要置信度）。",
    )
    observed_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("confidence")
    @classmethod
    def _confidence_only_for_inferred(cls, v: float | None, info) -> float | None:
        layer = info.data.get("layer")
        if layer == TraitLayer.OBSERVED and v is not None:
            raise ValueError("OBSERVED 层表示事实，不应携带 confidence")
        return v


class VisualProfile(BaseModel):
    """外貌特征。字段允许为空——VLM 抽不到时留空，**不得编造**。"""

    fur_color: str | None = None
    fur_length: str | None = None
    eye_color: str | None = None
    body_shape: str | None = None
    face_shape: str | None = None
    distinctive_features: list[str] = Field(
        default_factory=list, description="如 '左耳边缘小缺口'、'胸口白毛'"
    )

    def feature_terms(self) -> set[str]:
        """展平成词项集合，用于多图取交集与一致性比对。"""
        terms: set[str] = set()
        for value in (
            self.fur_color,
            self.fur_length,
            self.eye_color,
            self.body_shape,
            self.face_shape,
        ):
            if value:
                terms.add(value)
        terms.update(self.distinctive_features)
        return terms


class PetProfileDraft(BaseModel):
    """`pet_profile_analyzer` 节点的输出：待用户确认的档案草案。"""

    name: str | None = None
    breed: str | None = None
    visual: VisualProfile = Field(default_factory=VisualProfile)
    must_keep_features: list[str] = Field(
        default_factory=list,
        description="多图一致出现的特征。作为身份一致性的硬约束注入后续所有 prompt。",
    )
    observed_but_unstable: list[str] = Field(
        default_factory=list,
        description="仅在部分图片出现的特征。仅供参考，不作为身份约束。",
    )
    photo_count: int = 0
    analyzed_photo_count: int = 0
    failed_photo_urls: list[str] = Field(default_factory=list)
    coverage_note: str | None = Field(
        default=None,
        description="给用户看的覆盖度说明，如 '基于 4/5 张照片'。部分失败时必填。",
    )

    @property
    def coverage(self) -> float:
        if self.photo_count == 0:
            return 0.0
        return self.analyzed_photo_count / self.photo_count


class PetProfile(BaseModel):
    """持久化的宠物档案：本项目的核心数据资产。"""

    pet_id: str
    user_id: str
    name: str
    species: Species = Species.CAT
    breed: str | None = None

    visual: VisualProfile = Field(default_factory=VisualProfile)
    must_keep_features: list[str] = Field(
        default_factory=list,
        description="身份一致性的硬约束。任何生成内容都必须与此一致。",
    )
    observed_but_unstable: list[str] = Field(default_factory=list)

    traits: list[Trait] = Field(
        default_factory=list,
        description="性格 / 喜好 / 习惯 / 互动偏好，均带层级归属。",
    )

    appearance_embedding_id: str | None = Field(
        default=None,
        description=(
            "外观一致性校验向量的存储引用。"
            "注意：通用图像 embedding 只能判断外观是否相符，"
            "不能作为个体身份鉴定（见 docs/00-product.md §3.1）。"
        ),
    )

    identity_prompt: str = Field(
        default=(
            "This is the same real cat. Preserve its exact identity and "
            "distinctive markings. Never invent features absent from the profile."
        )
    )

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def traits_by_layer(self, layer: TraitLayer) -> list[Trait]:
        return [t for t in self.traits if t.layer == layer]

    def identity_block(self) -> str:
        """注入 prompt 的身份约束块。只包含硬约束，不含推测。"""
        lines = [f"名字：{self.name}"]
        if self.breed:
            lines.append(f"品种：{self.breed}")
        if self.must_keep_features:
            lines.append("必须保留的特征：" + "、".join(self.must_keep_features))
        return "\n".join(lines)
