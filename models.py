"""Domain models for the Belief Compatibility Mapper."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Belief(BaseModel):
    """A single belief with its raw text, LLM-expanded definition, and embedding."""

    id: int = Field(..., ge=0, lt=15, description="Unique belief index (0-14)")
    text: str = Field(..., min_length=1, description="Raw belief statement")
    expanded: str = Field(
        default="",
        description="LLM-generated expanded definition of the belief",
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Vector embedding of the belief (populated by an encoder)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Thematic tags for cluster-based pairing (e.g. 'ethics', 'economics')",
    )


class TensionCategory(str, Enum):
    """The five possible logical relationships between two beliefs."""

    ENTAILED = "mutually_entailed"
    HARMONIOUS = "compatible_harmonious"
    NEUTRAL = "neutral"
    TENSIONED = "tensioned"
    CONTRADICTORY = "contradictory"


class TensionResult(BaseModel):
    """Structured output from an LLM logical-tension analysis."""

    score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Compatibility score: -1.0 (contradictory) to +1.0 (entailed)",
    )
    category: TensionCategory = Field(
        ..., description="Which of the five relationship categories applies",
    )
    justification: str = Field(
        ..., min_length=1,
        description="One-sentence justification for the score",
    )


class BeliefRecommendation(BaseModel):
    """A suggested new belief returned by the recommend endpoint."""

    text: str = Field(..., min_length=1)
    justification: str = Field(..., min_length=1)


class BedrockPrinciple(BaseModel):
    """An implicit foundational principle that unifies two or more beliefs."""

    principle: str = Field(..., min_length=1)
    belief_ids: list[int] = Field(..., min_length=2)
    coherence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)


class DissonanceAlert(BaseModel):
    """A detected contradiction between two beliefs with downstream impact analysis."""

    belief_id_a: int = Field(..., ge=0, lt=15)
    belief_id_b: int = Field(..., ge=0, lt=15)
    score: float = Field(..., ge=-1.0, le=1.0)
    dependent_ids: list[int] = Field(
        default_factory=list,
        description=(
            "IDs of beliefs that positively align with A or B and are "
            "therefore at risk if A and B cannot both be true."
        ),
    )
    severity: float = Field(
        ..., ge=0.0, le=1.0,
        description="abs(score) * (1 + 0.1 * n_dependents), clamped to 1.0",
    )
