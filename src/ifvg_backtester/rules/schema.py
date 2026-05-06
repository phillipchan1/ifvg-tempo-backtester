"""Rule schema for versioned entry-model definitions.

Each rule is a structured object carrying its own hypothesis and a discrete
permutation space, so a permutation runner can vary one knob at a time while
holding the rest at their `default` values.
"""

from typing import Any, List, Literal
from pydantic import BaseModel, Field


RuleCategory = Literal[
    "session",
    "bias",
    "sweep",
    "fvg",
    "target_selection",
    "trigger",
    "execution",
    "risk",
    "trade_management",
    "filter",
    "timeframe",
]


class Rule(BaseModel):
    id: str = Field(description="Stable rule id, e.g. R01_session_window")
    category: RuleCategory
    description: str
    hypothesis: str = Field(
        description="Why we believe this rule contributes to edge — testable claim"
    )
    default: Any = Field(description="Value used when not permuting")
    permutation_space: List[Any] = Field(
        default_factory=list,
        description="Discrete alternative values to test. Empty if locked.",
    )
    locked: bool = Field(
        default=False,
        description="If true, the permutation runner must skip this rule.",
    )
    notes: str = ""


class EntryRuleSet(BaseModel):
    version: str
    name: str
    narrative_tags: List[str] = Field(default_factory=list)
    rules: List[Rule] = Field(default_factory=list)
    notes: str = ""
