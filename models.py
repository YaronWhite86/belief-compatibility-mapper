"""Domain models for the Belief Compatibility Mapper."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Belief(BaseModel):
    """A single belief with its raw text, LLM-expanded definition, and embedding."""

    id: int = Field(..., ge=0, lt=100, description="Unique belief index (0-99)")
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
