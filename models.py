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
