# IFVG A+ Reversal Model — Rule Spec v003

**Status:** v003. Tightens v002 with full pre-RTH sweep tracking. v002's lightweight "open-side" filter missed levels that were swept overnight then retraced back — e.g. 03-25 asia_high was wicked at 04:06 in London, but price retraced below it by NY open, so v002 kept it as fresh resistance. Three "winning" trades on 03-25 (cross-TF on 30s/1m/5m) were artifacts of that gap.

> A level we only care about is one that has NOT yet been swept. A wicked level is already taken liquidity — it can't be the reversal point.

**Companion:** [`configs/rules/ifvg_entry_v003.yaml`](../../configs/rules/ifvg_entry_v003.yaml) — machine-readable spec.

> **Versioning convention:** every version (v004, …) is a **complete standalone file**. Never a diff. Git history holds the diff; the file must always be readable as a self-contained spec for that point in time.

---

## 1. The setup in plain English (unchanged from v001)

> During the NY killzone (09:30–11:00 ET), price runs into a known liquidity pool — session H/L (London, PDH/PDL, Asia, Overnight, 6am, NWOG) or an HTF FVG zone (1H, 4H) — and sweeps it (any first-touch). On the reversal, price closes **through** an opposite-direction Fair Value Gap formed during the *RTH session itself*. That close-through is the **inverse FVG (IFVG)** trigger — entry on close, stop just past the far edge of the inverted gap, 1R take-profit.
>
> Bias is filtered by the prior-day midpoint: only shorts in premium, only longs in discount. Setup must be clean — at most 2 unfilled gaps in the way (think speed bumps); the most recent one is the target.

---

## 2. Rules

Each rule has: **description**, **hypothesis** (testable claim), **default** (used when not permuting), and **permutation space**.

---

### R01 — Session window
**Description.** Both sweep candle and inversion-trigger candle must close inside this US/Eastern window.

**Hypothesis.** 09:30–11:00 ET captures the highest-expectancy reversal-after-sweep window. After 11:00 the move tends to chop.

**Default.** `09:30 – 11:00 ET`
**Permutations.** `09:30–10:00`, `09:30–10:30`, `09:30–11:00`, `09:30–12:00`, `10:00–11:30`

---

### R02 — Bias reference range (premium/discount)
**Description.** Reference range whose 50% midpoint defines premium vs discount. Above 50% → premium → **shorts only**. Below 50% → discount → **longs only**.

**Hypothesis.** Prior-day range midpoint is the most stable bias filter for NY killzone reversals (matches the TradingView "Dynamic 50% Line" indicator).

**Default.** `prior_day`
**Permutations.** `prior_day`, `current_session`, `prior_week`, `last_24h`

---

### R03 — Liquidity target ⭐ EXPANDED in v002 (now swing-based)
**Description.** Which liquidity level(s) qualify as a valid sweep target. v002 default expands the pool to the full level taxonomy (matching the fvgc-backtest sister project) and changes session H/L definitions from raw extremes to **swing points**.

| Level type | Window | H/L definition |
|---|---|---|
| `prev_day` | Prior trading day RTH 09:30–16:00 NY | swing high/low |
| `london` | 02:00–08:00 NY (today) | swing high/low |
| `asia` | Prior 19:00 → today 02:00 NY | swing high/low |
| `overnight` | Prior 18:00 → today 09:30 NY | swing high/low |
| `6am` | 04:00–08:00 NY (today) | swing high/low |
| `nwog` | Friday RTH close ↔ Monday RTH open (Mondays only) | two prices, no swings |
| `htf_fvg_1H` | 1-hour FVGs, near edge as the sweep level | n/a (gap zone) |
| `htf_fvg_4H` | 4-hour FVGs, near edge as the sweep level | n/a (gap zone) |

**Swing definition (v002 NEW):** within each session window, 1m bars are resampled to **5m**, and a bar at index `i` is a swing high if its high is the strict max of `[i-2, i+2]` (and similarly for lows). The session "high" is the **highest swing high** in the window; "low" is the **lowest swing low**.

> **Why:** in the v001 March 2026 run, asia_low for 03-11 was 24963.75 — a single 1m bar at 19:02, the third bar of the entire 7-hour session. Bars before and after had higher lows (24966.50, 24966.25). It was opening-tick noise, not a structural low. With swing-based detection, asia_low becomes 24975.00 (a 5m swing low at 19:35) — a real pivot visible on a chart.

**Permutation space for swing detection:** swing TF in `{1min, 3min, 5min, 15min}`; N in `{2, 3, 5}`. Default is `15m / N=2` — matches the timeframe a discretionary trader eyeballs for "Asia low," "London high," etc. 5m and lower introduce noise pivots that aren't visible on a typical chart.

**Pre-RTH sweep filter (v003 NEW):** at session start, each level is checked against the pre-RTH bars in the window AFTER its formation ended. If price wicked the level at any point (any bar's high ≥ resistance, any bar's low ≤ support), the level is **excluded** — it's already taken liquidity.

Per-group sweep-check windows (matches fvgc-backtest's `_finalize_pre_rth_sweep`):

| Group | Formation ends | Sweep check window |
|---|---|---|
| `prev_day` | Prior RTH close | Whole pre-RTH window (entire overnight) |
| `nwog` | Friday RTH close | Whole pre-RTH window |
| `asia` | 02:00 NY | 02:00 → 09:30 |
| `london` | 08:00 NY | 08:00 → 09:30 |
| `6am` | 08:00 NY | 08:00 → 09:30 |
| `bsl_ssl` | 08:00 NY | 08:00 → 09:30 |
| `overnight` | (IS the overnight extreme) | Always considered fresh |
| `htf_fvg_*` | 3rd-candle close | Tracked via close-through inversion at level-build time |

> **Why:** an `asia_high` formed at 23:00 the night before, then swept by a London wick at 04:06, is no longer a fresh resistance pool — even if price retraced back below by 09:30. The liquidity was taken at 04:06. The setup mechanically requires "price runs into untouched liquidity, then reverses." Already-swept levels violate that premise.

**Hypothesis.** Swing points are real liquidity pools that traders watch; opening ticks aren't. A wider level set produces more setups; per-group analysis lets us learn which level types actually generate edge.

**Default.** All eight types listed above. Swing detection: `swing_tf=15min, swing_n=2`.
**Permutations.** Various subsets of level types; swing detection params (see YAML).

---

### R04 — Sweep validity
**Description.** A sweep is the **first session bar** where price reaches the level — wick OR close, no distinction. R09's inversion close-through does the actual reversal filtering.

**Hypothesis.** Filtering reversal at the sweep candle double-counts what R09 already requires.

**Default.** `first_touch`
**Permutations.** `first_touch`, `wick_only_with_reversal`, `close_through_then_reclaim`

---

### R05 — FVG definition ⭐ CHANGED in v002
**Description.** Standard 3-candle FVG.
- **Bullish FVG:** `candle[i-2].high < candle[i].low` → gap = `[c[i-2].high, c[i].low]`
- **Bearish FVG:** `candle[i-2].low > candle[i].high` → gap = `[c[i].high, c[i-2].low]`
- Middle candle (`i-1`) is the displacement leg.

**v002 NEW — eligibility window:** only FVGs formed during the RTH session (`formed_at ≥ 09:30 NY` on the session date) are eligible as inversion targets. **Pre-market FVGs are excluded.**

> **Why:** in March 2026, the 03-10 short trade selected a 7.2pt 1m bullish FVG that formed at 09:24 (pre-market) as its inversion target. Pre-market gaps reflect overnight/Asian/London structure, not the killzone reversal context. The trade lost — predictable, since the target wasn't part of the morning displacement.

**Default.**
```
shape: wick_fvg
eligibility_window: rth_only        ← v002 new
```
**Permutations.** `{rth_only, same_day}` for the eligibility window; `{wick_fvg, body_fvg, wick_fvg_with_displacement_strength}` for shape. `same_day` reproduces v001 behavior.

---

### R06 — Target gap size (points)
**Description.** Acceptable size of the target gap in instrument points (NQ defaults). Gaps outside this band disqualify the setup.

**Hypothesis.** 7–20 point gaps on NQ are large enough to matter structurally but small enough to invert quickly inside the killzone.

**Default.** `[7, 20]`
**Permutations.** `[3, 7]`, `[7, 12]`, `[12, 20]`, `[20, 50]`, `[3, 50]`, `[5, 30]`

---

### R07 — Target gap selection ⭐ CHANGED in v002
**Description.** Picks **which** gap to "watch" for inversion after the sweep.

**Eligibility (universal hard filters):**
- Opposite-direction unfilled FVG.
  - Short setup: **bullish** FVG with `gap.low < current_price`.
    - **v002 NEW**: gaps STRADDLING current price are now eligible — the gap's top can sit *above* current price as long as the bottom is below. Close-through still fires when `close < gap.low`.
  - Long setup: **bearish** FVG with `gap.high > current_price`.
- Size in R06 band.
- Formed during the RTH session (R05 `eligibility_window=rth_only`).
- Not already inverted (close-through) by any prior bar.
- Near edge within `eligibility_proximity_points` of the swept level.

**Selection (after eligibility filter):**

| Count | Action |
|---|---|
| 0 | **SKIP** — nothing to inverse. |
| 1 | Take it. |
| 2 | Target = `target_when_multiple` (default `most_recent`). |
| > `max_gaps` | **SKIP** — "too many speed bumps." |

**Mental model.** Each unfilled gap in the recent rally is a speed bump. 1–2 bumps clean. 3+ in the rally context = dirty setup, skip.

> **Why the straddling fix:** on 03-04, the 4H FVG sweep at 10:36 had a 1m bullish FVG formed at 10:27 (top 25112, bot 25095). Sweep close was 25106 — between the gap's top and bottom. The v001 filter required `gap.high <= close`, which rejected this perfectly valid inversion target. The setup we *did* select (a 10:22 gap further away) lost. The 10:27 setup would have won. This straddling pattern is common: the same displacement candle that sweeps an HTF level creates the inversion gap.

**Default.**
```
max_gaps: 2
eligibility_proximity_points: 50
target_when_multiple: most_recent
allow_straddling_gaps: true        ← v002 new
```

**Permutations** (rows):
- `max_gaps=1, prox=25, most_recent, straddling=true`
- `max_gaps=2, prox=50, most_recent, straddling=true`  ← default
- `max_gaps=2, prox=30, most_recent, straddling=true`
- `max_gaps=2, prox=75, most_recent, straddling=true`
- `max_gaps=3, prox=75, most_recent, straddling=true`
- `max_gaps=2, prox=50, closest_to_swept_level, straddling=true`
- `max_gaps=2, prox=50, most_recent, straddling=false`  ← reproduces v001 strict-below

---

### R08 — Time-to-inversion (minutes) ⭐ CHANGED to time-based in v002 calibration
**Description.** Max wall-clock minutes between the sweep candle and the inversion-trigger candle. Beyond this the setup expires and the engine resets. **Time-based, not bar-based** — same logic firing on 30s and 5m bars must allow the same real-time window for the move to develop.

**Hypothesis.** "No stalling" — fast inversion signals strong reversal intent. 15 min ≈ one ICT macro window (e.g. 09:30–9:45) and is the longest reasonable wait before the structural reversal premise becomes stale.

**Why time-based:** the v001/v002 bar-based default (10 bars) produced absurd ranges across TFs — 5 min on 30s, 50 min on 5m. A real reversal takes the same wall-clock time regardless of what TF you sample at. Multi-TF run on March 2026 confirmed the bar-based version was the dominant difference between TF outcomes.

**Default.** `15` minutes
**Permutations.** `5`, `10`, `15`, `20`, `30` minutes

---

### R09 — Inversion trigger
**Description.** What counts as "inverting" the target gap.
- **Short:** candle **closes below** `target_gap.low`.
- **Long:** candle **closes above** `target_gap.high`.
- Body-pct variants additionally require `body / range ≥ threshold`.

**Default.** `close_through`
**Permutations.** `wick_through`, `close_through`, `close_through_body_pct_50`, `close_through_body_pct_70`

---

### R10 — Entry execution
**Description.** How and when the order fills once the trigger condition is met.

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

**Default.** `buffer_points: 5`
**Permutations.** `0`, `5`, `10`, `15` points

---

### R12 — Take profit
**Description.** Profit-taking and trade-management policy.

**Default.** `full_at_1R`
**Permutations.** `full_at_1R`, `full_at_2R`, `half_1R_runner_to_swing`, `full_at_nearest_swing`, `swing_to_swing`

---

### R13 — Re-entry policy
**Description.** How many trades allowed per session per direction.

**Default.** `one_per_session_per_direction`
**Permutations.** `one_per_session_per_direction`, `two_per_session_per_direction`, `unlimited_per_session`

---

### R14 — News filter
**Description.** Skip days where a high-impact US economic event lands inside the session window.

**Default.** `us_red_folder`
**Permutations.** `off`, `us_red_folder`, `cpi_fomc_nfp_only`

---

### R15 — Execution timeframe 🔒 LOCKED
**Description.** Bar timeframe on which sweep, FVG, and inversion are detected.

**Default.** `1m`
**Permutations.** *(none — locked)*

---

## 3. v003 vs v002 — what changed

| | v002 | v003 |
|---|---|---|
| R03 pre-RTH sweep filter | **lightweight** ("level on wrong side of session open") | **full** time-window check after formation: any wick that touched the level marks it swept |

Single-rule change. Same level pool, same swing-based H/L (15m N=2), same time-based inversion (15 min), same RTH-only inversion targets, same straddling-gap fix. Only the *availability* of each level at session start gets stricter.

**Expected impact:** 2–5 fewer trades on March 2026; the 03-25 cross-TF "triple winner" on asia_high should drop entirely.

### Full history v001 → v003

| | v001 | v002 | v003 |
|---|---|---|---|
| R03 sweep targets | `[prev_day, london]` | All 8 types | (same) |
| R03 session H/L definition | raw min/max | swing-based (15m N=2) | (same) |
| R03 pre-RTH sweep | n/a | lightweight open-side | **full time-window** |
| R05 inversion target window | any (incl. pre-market) | RTH-only | (same) |
| R07 straddling gaps | rejected | accepted | (same) |
| R08 time-to-inversion | bar-based 10 | **time-based 15 min** | (same) |

---

## 4. Permutation discipline (unchanged)

Sweep one rule at a time, holding the rest at their defaults. Suggested order after v002 baseline:
1. R03 (sweep target subsets) — most consequential lever; learn which level types produce edge.
2. R07 (proximity, max_gaps, straddling toggle).
3. R08 (time-to-inversion).
4. R09 (trigger strength).
5. R06 (gap size band).
6. R11 (stop buffer).
7. R12 (take profit).
8. R13 (re-entry).

---

## 5. What's NOT in v002 (deferred)

- BSL/SSL (London 15m swing H/L) — needs swing-point detector. Easy to add; low priority per fvgc-backtest analysis (60% WR small sample).
- Volume profile levels (POC/VAH/VAL).
- Multi-TF execution (30s, 2m, 3m, 5m). Locked to 1m.
- Walk-forward train/test split. Required before any conclusions.
- Slippage / commissions. Engine first, costs second.
- Position sizing. Unit-R for all permutations.
