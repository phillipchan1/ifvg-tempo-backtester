# Rule Versioning

Two paired files per version, both **complete and standalone** (never a diff):

- `configs/rules/ifvg_entry_vNNN.yaml` — machine-readable spec consumed by the engine.
- `docs/rules/ifvg_entry_vNNN.md` — plain-English narrative of the same rules.

## Convention

- Every new version is a **complete file**, not a diff. Copy the previous version's pair, bump the version, and edit in place.
- Git history is where diffs live. The file itself must always read as a self-contained spec for that point in time.
- Keep YAML and MD in sync — the YAML is authoritative for the engine; the MD is authoritative for human review. They describe the same rules.

## Naming

- `ifvg_entry_v001.yaml` / `ifvg_entry_v001.md`
- `ifvg_entry_v002.yaml` / `ifvg_entry_v002.md`
- …

## Each rule carries

1. `id` — stable identifier (e.g. `R07_target_gap_selection`)
2. `category` — one of: `session`, `bias`, `sweep`, `fvg`, `target_selection`, `trigger`, `execution`, `risk`, `trade_management`, `filter`, `timeframe`
3. `description` — what the rule does, mechanically
4. `hypothesis` — testable claim about why this rule contributes to edge
5. `default` — value used when not permuting
6. `permutation_space` — discrete alternative values to test (empty if locked)
7. `locked` — boolean; if true, the permutation runner skips this rule

## Permutation discipline

Sweep **one rule at a time**, holding the rest at their defaults. Changing multiple knobs in the same run makes attribution impossible.
