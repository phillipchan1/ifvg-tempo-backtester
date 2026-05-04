"""Rule schema for versioned entry model definitions."""

from pydantic import BaseModel, Field
from typing import List


class EntryRuleSet(BaseModel):
    version: str = Field(description="Semantic or date version for this rule set")
    name: str = Field(description="Human-readable rule set name")
    narrative_tags: List[str] = Field(default_factory=list)
    setup_requirements: List[str] = Field(default_factory=list)
    invalidation_rules: List[str] = Field(default_factory=list)
    execution_rules: List[str] = Field(default_factory=list)
    risk_rules: List[str] = Field(default_factory=list)
    notes: str = ""
