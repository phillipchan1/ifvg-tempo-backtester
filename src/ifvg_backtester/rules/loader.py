"""Helpers for loading versioned rule files from configs/rules."""

from pathlib import Path
import yaml

from .schema import EntryRuleSet


def load_rule_file(rule_path: str | Path) -> EntryRuleSet:
    path = Path(rule_path)
    payload = yaml.safe_load(path.read_text())
    return EntryRuleSet.model_validate(payload)
