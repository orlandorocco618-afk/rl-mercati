# Strategy Execution TODO

## Goal
Build a stronger SR-first research and execution pipeline, then test whether BR adds value only in the regimes where it truly helps.

## Phase 1 - Diagnostics
- [x] Rebuild BR vs SR OOS comparison dashboard
- [x] Compare best single models and weighted portfolios
- [x] Add monthly-by-regime diagnostics for BR and SR
- [x] Measure where SR dominates and where BR still adds edge

## Phase 2 - Regime Selector
- [x] Build an adaptive SR/BR selector driven by trailing monthly regime performance
- [x] Compare selector vs SR best vs BR best vs static weighted portfolios
- [x] Count selector switches and isolate regime-specific decision quality
- [x] Add realistic switching-cost assumptions

## Phase 3 - Better Objective
- [x] Stop optimizing only `total_return_pct`
- [x] Optimize for monthly consistency: median month, worst month, drawdown, turnover
- [x] Track net performance after spread/slippage stress
- [x] Add walk-forward evaluation windows instead of a single OOS summary
- [x] Extend training defaults with drawdown-aware and confidence-aware reward terms

## Phase 4 - Quality Filters
- [x] Add confidence / no-trade filter — tested, discarded (see the conclusion)
- [x] Reject low-edge bars even when the policy emits an action — discarded together with the gate
- [x] Measure expectancy improvement vs activity reduction — the gate cuts trades from 95 to 41 without improving the net return
- [x] Evaluate confidence gate overlays on `SR_Robust`
- [x] Promote the gate only if it beats pure `SR` on net OOS return

**Phase 4 conclusion:** gate NOT promoted.
- Baseline: +1.74% OOS, 95 trades
- Dynamic Size (no gate): +1.94% OOS, 95 trades — WINNER
- Dynamic + Gate 0.62: +1.66% OOS, 41 trades
- Dynamic + Gate 0.58: +1.52% OOS, 51 trades
- Overlay to use on SR_Robust: Dynamic Size only, no gate.

## Phase 5 - Position Sizing
- [x] Add dynamic sizing by volatility / ATR / regime confidence
- [x] Compare baseline vs dynamic sizing overlays on `SR_Robust`
- [ ] Cap exposure during weak or unstable regimes
- [ ] Retrain `SR` with the new sizing-aware objective and compare against the old baseline
- [ ] Stress test size rules under worse slippage assumptions

## Phase 6 - Opportunity Set
- [ ] Expand beyond the current single-strategy/single-context setup
- [ ] Test additional assets and/or timeframes
- [ ] Verify whether the edge improves from more opportunities rather than more leverage
- [ ] Choose together between the first two expansion paths: `multi-asset XAU-like basket` vs `multi-timeframe XAU`

## Phase 7 - Promotion Rules
- [ ] Define minimum standards for promotion
- [ ] Suggested first threshold: stable positive OOS with controlled drawdown
- [ ] Only after that, push for higher monthly returns
