# IFVG A+ Reversal Model — Rule Spec v001

**Status:** First rigorous spec (replaces the prose-only earlier draft).
**Scope of first run:** 1-minute timeframe, ~2 weeks of NQ data, engine smoke-test only.
**Companion:** [`configs/rules/ifvg_entry_v001.yaml`](../../configs/rules/ifvg_entry_v001.yaml) — machine-readable form of these rules.

> **Versioning convention:** every future version (v002, v003, …) is a **complete standalone file**. Never a diff. Git history holds the diff; the file must always be readable as a self-contained spec.

---

## 1. The setup in plain English

> During the NY killzone (09:30–11:00 ET), price runs into a known liquidity pool (London H/L or PDH/PDL), sweeps it with at least a wick, and reverses. On the reversal, price closes **through** an opposite-direction Fair Value Gap that exists between current price and the swept level. That close-through is the **inverse FVG (IFVG)** trigger — we enter at the close of that candle, with a stop just past the far edge of the inverted gap.
>
> Bias is filtered by the prior-day midpoint: only shorts in premium, only longs in discount. Setup must be clean — at most 2 unfilled gaps in the way (think speed bumps); the most recent one is the target.

---

## 2. Rules

Each rule has: **description**, **hypothesis** (what we believe and why — a testable claim), **default** (used when not permuting), and **permutation space** (alternatives we'll sweep one knob at a time).

---

### R01 — Session window
**Description.** Both the sweep candle and the inversion-trigger candle must close inside this US/Eastern window. Setups straddling the boundary are invalid.

**Hypothesis.** 09:30–11:00 ET captures the highest-expectancy reversal-after-sweep window. After 11:00 the move tends to chop.

**Default.** `09:30 – 11:00 ET`
**Permutations.** `09:30–10:00`, `09:30–10:30`, `09:30–11:00`, `09:30–12:00`, `10:00–11:30`

---

### R02 — Bias reference range (premium/discount)
**Description.** Reference range whose 50% midpoint defines premium vs discount. Above 50% → premium → **shorts only**. Below 50% → discount → **longs only**.

**Hypothesis.** Prior-day range midpoint is the most stable and widely-watched bias filter for NY killzone reversals (matches the TradingView "Dynamic 50% Line" indicator).

**Default.** `prior_day`
**Permutations.** `prior_day`, `current_session`, `prior_week`, `last_24h`

---

### R03 — Liquidity target
**Description.** Which liquidity level(s) qualify as a valid sweep target. The setup needs at least one of these to be swept (per R04) before the trigger.

**Hypothesis.** London H/L and prior-day H/L are the most reliable pools for the 09:30 ET reversal. Asia is too stale; intraday swings too noisy.

**Default.** `[london_high_low, prior_day_high_low]`
**Permutations.**
- `[asia_high_low]`
- `[london_high_low]`
- `[prior_day_high_low]`
- `[london_high_low, prior_day_high_low]`
- `[prior_session_swing]`
- `[htf_fvg_4h]`
- `[htf_fvg_1h]`

---

### R04 — Sweep validity
**Description.** A sweep is the **first session bar** in which price reaches the level — wick OR close, no distinction. "Swept" and "broken" are the same thing here. The R09 inversion-close trigger does the actual reversal filtering.

**Hypothesis.** Filtering reversal at the sweep candle double-counts what R09 already requires. Demanding a same-candle reversal over-fits to specific candle shapes and discards setups that legitimately reverse a few bars after first touch.

**Default.** `first_touch`
**Permutations.** `first_touch`, `wick_only_with_reversal`, `close_through_then_reclaim`

---

### R05 — FVG definition
**Description.** Standard 3-candle FVG.
- **Bullish FVG:** `candle[i-2].high < candle[i].low` → gap = `[c[i-2].high, c[i].low]`
- **Bearish FVG:** `candle[i-2].low > candle[i].high` → gap = `[c[i].high, c[i-2].low]`
- The middle candle (`i-1`) is the displacement leg.

**Hypothesis.** Wick-based FVG is the canonical ICT definition. Body-based FVGs and displacement-strength filters trade sample size for setup quality.

**Default.** `wick_fvg`
**Permutations.** `wick_fvg`, `body_fvg`, `wick_fvg_with_displacement_strength`

---

### R06 — Target gap size (points)
**Description.** Acceptable size of the target gap, in instrument points (NQ defaults). Gaps outside this band disqualify the setup.

**Hypothesis.** 7–20 point gaps on NQ are large enough to matter structurally but small enough to invert quickly inside the killzone.

**Default.** `[7, 20]`
**Permutations.** `[3, 7]`, `[7, 12]`, `[12, 20]`, `[20, 50]`, `[3, 50]` (control: any size)

---

### R07 — Target gap selection ⭐
**Description.** Picks **which** gap to "watch" for inversion after the sweep.

**Eligible gaps** (universal hard filters):
- Opposite-direction unfilled FVG between current price and swept level
  - Short setup (high sweep) → bullish FVGs **below** price.
  - Long setup (low sweep) → bearish FVGs **above** price.
- Size in R06 band.
- Formed within `eligibility_lookback_bars` of the sweep — captures only gaps from the **rally to the sweep**.
- Near edge within `eligibility_proximity_points` of the swept level.

**Selection** (after eligibility filter):

| Count | Action |
|---|---|
| 0 | **SKIP** — nothing to inverse. |
| 1 | Take it. |
| 2 | Target = `target_when_multiple` (default `most_recent`). |
| > `max_gaps` | **SKIP** — "too many speed bumps." |

**Mental model.** Each unfilled gap in the immediate rally is a speed bump. 1–2 bumps = clean. 3+ in the rally context = dirty setup, skip.

**Hypothesis.** Cleaner setups (1–2 recent, near-the-sweep gaps) reverse with less retracement and higher follow-through.

**Default.**
```
max_gaps: 2
eligibility_lookback_bars: 30        # 30 min on 1m
eligibility_proximity_points: 50     # NQ points
target_when_multiple: most_recent
```

**Permutations** (each row = one full configuration):
- `max_gaps=1, lookback=30, prox=50, most_recent`
- `max_gaps=2, lookback=30, prox=50, most_recent`  (default)
- `max_gaps=2, lookback=60, prox=50, most_recent`
- `max_gaps=2, lookback=30, prox=30, most_recent`
- `max_gaps=2, lookback=30, prox=75, most_recent`
- `max_gaps=3, lookback=60, prox=75, most_recent`
- `max_gaps=2, lookback=30, prox=50, closest_to_swept_level`

---

### R08 — Time-to-inversion (candles)
**Description.** Max number of candles allowed between the sweep candle and the inversion-trigger candle. Beyond this, the setup expires and the engine resets.

**Hypothesis.** "No stalling" — fast inversion signals strong reversal intent. Slow inversions are usually grind/chop. 10 bars (10 min on 1m) gives enough room for a real reversal to form without admitting drifting setups.

**Default.** `10`
**Permutations.** `3`, `5`, `10`, `15`, `20`

---

### R09 — Inversion trigger
**Description.** What counts as "inverting" the target gap.
- **Short:** candle **closes below** `target_gap.low`.
- **Long:** candle **closes above** `target_gap.high`.
- Body-pct variants additionally require `body / range ≥ threshold`.

**Hypothesis.** A close-through is the minimum signal. Body-strength filters approximate the chart's "strong momentum close" rule and should improve win-rate at the cost of sample size.

**Default.** `close_through`
**Permutations.** `wick_through`, `close_through`, `close_through_body_pct_50`, `close_through_body_pct_70`

---

### R10 — Entry execution
**Description.** How and when the order fills once the trigger condition is met.

**Hypothesis.** Market on close of the inversion candle is the simplest and most reproducible in backtest. Limit at gap mid is most aggressive (better avg price, worse fill rate).

**Default.** `market_on_inversion_close`
**Permutations.** `market_on_inversion_close`, `market_on_next_candle_open`, `limit_at_gap_mid`

---

### R11 — Stop loss
**Description.** Absolute stop-loss level. R per trade is intentionally **variable** (gap-size dependent).

```
Short: stop = target_gap.high + buffer_points
Long:  stop = target_gap.low  − buffer_points
R     = |stop − entry|
```

**Hypothesis.** The far edge of the inverted gap is the structural invalidation point. A small buffer absorbs noise without materially hurting R.

**Default.** `buffer_points: 5`
**Permutations.** `0`, `5`, `10`, `15` points

---

### R12 — Take profit
**Description.** Profit-taking and trade-management policy.

**Hypothesis.** 1R is the cleanest baseline for evaluating raw signal quality. Runners capture fat tails when momentum continues into a clear swing.

**Default.** `full_at_1R`
**Permutations.** `full_at_1R`, `full_at_2R`, `half_1R_runner_to_swing`, `full_at_nearest_swing`, `swing_to_swing`

---

### R13 — Re-entry policy
**Description.** How many trades allowed per session per direction.

**Hypothesis.** First setup of the session has the cleanest narrative. Re-entries usually dilute edge — but should be measured rather than assumed bad.

**Default.** `one_per_session_per_direction`
**Permutations.** `one_per_session_per_direction`, `two_per_session_per_direction`, `unlimited_per_session`

---

### R14 — News filter
**Description.** Skip days where a high-impact US economic event lands inside the session window.

**Hypothesis.** Event-driven volatility violates the structural reversal premise; the move becomes news-driven, not liquidity-driven.

**Default.** `us_red_folder`
**Permutations.** `off`, `us_red_folder`, `cpi_fomc_nfp_only`

---

### R15 — Execution timeframe 🔒 LOCKED
**Description.** Bar timeframe on which sweep, FVG, and inversion are detected.

**Hypothesis.** 1m is the right granularity for NY killzone reversal scalps. Locked for v001 to contain degrees of freedom during initial validation.

**Default.** `1m`
**Permutations.** *(none — locked)*

---

## 3. Permutation discipline

When v001 detectors are validated, the runner sweeps **one rule at a time**, holding all others at their `default`. This is non-negotiable: changing two knobs at once makes the result attribution ambiguous.

Order of permutations (suggested):
1. R07 (target gap selection) — biggest mechanical lever, unique to this model.
2. R08 (time-to-inversion) — tightens "intent."
3. R09 (inversion trigger strength) — quality vs sample tradeoff.
4. R06 (gap size band) — structural cleanliness.
5. R11 (stop buffer) — risk geometry.
6. R02 (bias reference range) — context layer.
7. R12 (take profit) — exit policy is best evaluated after entry edge is confirmed.
8. R13 (re-entry) — last; only meaningful once base expectancy is positive.

---

## 4. What's NOT in v001 (deferred deliberately)

- Multi-timeframe permutation (30s, 2m, 3m, 5m). Locked to 1m to prevent overfitting via TF-shopping. Re-introduce once edge is confirmed on 1m and we want **out-of-sample TF confirmation**, not tuning.
- Walk-forward train/test split. Out of scope for the smoke-test phase. Required before any conclusions.
- Slippage and commission models. Engine first, costs second.
- Position sizing. All sizing is unit-R for v001.
