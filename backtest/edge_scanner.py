import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from utils.data_utils import merge_h1_with_d1_regimes
from utils.indicators import rsi
from utils.kalman import apply_kalman_filter


def _parse_breakout_lookbacks_from_env(default: list[int] | None = None) -> list[int]:
    defaults = default or [72, 96, 144]
    raw = os.getenv("RL_BREAKOUT_LOOKBACKS", "").strip()
    if not raw:
        return sorted(set(int(x) for x in defaults if int(x) >= 4))
    values = []
    for token in raw.replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        try:
            v = int(t)
        except Exception:
            continue
        if v >= 4:
            values.append(v)
    if not values:
        return sorted(set(int(x) for x in defaults if int(x) >= 4))
    return sorted(set(values))


def _split_train_test_by_date(
    df: pd.DataFrame,
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
):
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")
    out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    if val_end_ts <= train_end_ts:
        raise ValueError("val_end must come after train_end")
    df_train = out.loc[out["Time"] <= train_end_ts].reset_index(drop=True)
    df_test = out.loc[out["Time"] > val_end_ts].reset_index(drop=True)
    return df_train, df_test


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out["Open"] = pd.to_numeric(out["Open"], errors="coerce")
    out["RET_1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["RET_4"] = out["Close"].pct_change(4)
    out["RET_24"] = out["Close"].pct_change(24)
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["STD_20"] = out["Close"].rolling(20).std()
    out["Z_20"] = (out["Close"] - out["SMA_20"]) / (out["STD_20"] + 1e-9)
    out["RSI_14"] = rsi(out["Close"]).clip(0, 100) / 100.0
    atr_col = "ATR_H1" if "ATR_H1" in out.columns else ("ATR" if "ATR" in out.columns else None)
    if atr_col is not None:
        out["ATR_NORM"] = pd.to_numeric(out[atr_col], errors="coerce") / (out["Close"].abs() + 1e-9)
    else:
        out["ATR_NORM"] = np.nan
    out["VOL_Z_50"] = (
        (pd.to_numeric(out.get("TickVolume", 0.0), errors="coerce") - pd.to_numeric(out.get("TickVolume", 0.0), errors="coerce").rolling(50).mean())
        / (pd.to_numeric(out.get("TickVolume", 0.0), errors="coerce").rolling(50).std() + 1e-9)
    )
    out["TREND_STRENGTH"] = out["RET_24"] / (out["RET_1"].rolling(24).std() + 1e-9)
    return out


def _rank_features(df_train: pd.DataFrame, horizons=(1, 4, 8, 24)) -> pd.DataFrame:
    feature_cols = [
        "RET_1",
        "RET_4",
        "RET_24",
        "Z_20",
        "RSI_14",
        "ATR_NORM",
        "VOL_Z_50",
        "TREND_STRENGTH",
    ]
    rows = []
    for h in horizons:
        target = np.log(df_train["Close"].shift(-h) / df_train["Close"])
        for feat in feature_cols:
            x = pd.to_numeric(df_train[feat], errors="coerce")
            valid = x.notna() & target.notna() & np.isfinite(x) & np.isfinite(target)
            n = int(valid.sum())
            if n < 500:
                continue
            xv = x[valid]
            yv = target[valid]
            spearman = float(xv.corr(yv, method="spearman"))
            pearson = float(xv.corr(yv, method="pearson"))
            dir_edge = float((np.sign(xv) * yv).mean())
            dir_std = float((np.sign(xv) * yv).std(ddof=1)) if n > 1 else 0.0
            t_stat = float((dir_edge / (dir_std + 1e-12)) * np.sqrt(max(1, n))) if dir_std > 0 else 0.0
            score = float(abs(spearman) + abs(pearson) + 0.01 * abs(t_stat))
            rows.append(
                {
                    "feature": feat,
                    "horizon_h": int(h),
                    "n": n,
                    "spearman": spearman,
                    "pearson": pearson,
                    "dir_edge": dir_edge,
                    "t_stat": t_stat,
                    "score": score,
                }
            )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out


def _cost_rate(row: pd.Series, spread_bps: float, slippage_bps: float, slippage_atr_coef: float) -> float:
    px = float(pd.to_numeric(row.get("Close"), errors="coerce"))
    atr_v = float(pd.to_numeric(row.get("ATR_H1", row.get("ATR", 0.0)), errors="coerce"))
    atr_norm = atr_v / (abs(px) + 1e-9) if np.isfinite(atr_v) else 0.0
    return max(0.0, (spread_bps + slippage_bps) / 10000.0 + slippage_atr_coef * max(0.0, atr_norm))


def _simulate_from_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    commission: float = 0.0002,
    spread_bps: float = 2.0,
    slippage_bps: float = 1.0,
    slippage_atr_coef: float = 0.0,
    execution_delay_bars: int = 1,
):
    data = df.reset_index(drop=True).copy()
    sig = pd.to_numeric(signal, errors="coerce").fillna(0).clip(-1, 1).astype(int).reset_index(drop=True)
    n = len(data)
    if n <= execution_delay_bars + 2:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "total_trades": 0, "pct_profitable": 0.0, "expectancy": 0.0, "active_steps_pct": 0.0}

    equity_realized = 10000.0
    equity_curve = []
    position = 0
    entry_price = None
    trades = []
    active = 0

    for t in range(0, n - execution_delay_bars):
        intent = int(sig.iloc[t])
        exec_idx = t + execution_delay_bars
        row_dec = data.iloc[t]
        row_exec = data.iloc[exec_idx]
        px = pd.to_numeric(row_exec.get("Open", row_exec.get("Close")), errors="coerce")
        if pd.isna(px):
            close_t = float(pd.to_numeric(row_dec.get("Close"), errors="coerce"))
            if position == 1 and entry_price is not None:
                mtm = equity_realized * (1.0 + (close_t - entry_price) / (entry_price + 1e-9))
            elif position == -1 and entry_price is not None:
                mtm = equity_realized * (1.0 + (entry_price - close_t) / (entry_price + 1e-9))
            else:
                mtm = equity_realized
            equity_curve.append(float(mtm))
            continue
        px = float(px)
        c = _cost_rate(row_dec, spread_bps=spread_bps, slippage_bps=slippage_bps, slippage_atr_coef=slippage_atr_coef)
        buy_fill = px * (1.0 + c)
        sell_fill = px * (1.0 - c)
        realized = 0.0

        if intent != 0:
            active += 1

        if intent == 1 and position <= 0:
            if position == -1 and entry_price is not None:
                realized += (entry_price - buy_fill) / (entry_price + 1e-9)
                trades.append(realized - commission)
                realized -= commission
            position = 1
            entry_price = buy_fill
            realized -= commission
        elif intent == -1 and position >= 0:
            if position == 1 and entry_price is not None:
                realized += (sell_fill - entry_price) / (entry_price + 1e-9)
                trades.append(realized - commission)
                realized -= commission
            position = -1
            entry_price = sell_fill
            realized -= commission

        equity_realized += realized * equity_realized

        close_t = float(pd.to_numeric(row_dec.get("Close"), errors="coerce"))
        if position == 1 and entry_price is not None and np.isfinite(close_t):
            mtm = equity_realized * (1.0 + (close_t - entry_price) / (entry_price + 1e-9))
        elif position == -1 and entry_price is not None and np.isfinite(close_t):
            mtm = equity_realized * (1.0 + (entry_price - close_t) / (entry_price + 1e-9))
        else:
            mtm = equity_realized
        equity_curve.append(float(mtm))

    if position != 0 and entry_price is not None:
        last_px = float(pd.to_numeric(data.iloc[-1].get("Close"), errors="coerce"))
        if position == 1:
            tr = (last_px - entry_price) / (entry_price + 1e-9) - commission
        else:
            tr = (entry_price - last_px) / (entry_price + 1e-9) - commission
        trades.append(tr)
        equity_realized += tr * equity_realized
        equity_curve.append(float(equity_realized))

    eq = np.array(equity_curve, dtype=float) if equity_curve else np.array([10000.0], dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / (running_max + 1e-9) * 100.0
    ret = 100.0 * (eq[-1] - 10000.0) / 10000.0
    win_rate = 100.0 * sum(1 for x in trades if x > 0) / max(1, len(trades))
    expectancy = float(np.mean(trades)) if trades else 0.0
    return {
        "total_return_pct": float(ret),
        "max_drawdown_pct": float(np.min(dd)),
        "total_trades": int(len(trades)),
        "pct_profitable": float(win_rate),
        "expectancy": float(expectancy),
        "active_steps_pct": float(100.0 * active / max(1, len(sig))),
    }


def _make_signal(df: pd.DataFrame, family: str, params: dict) -> pd.Series:
    if family == "momentum":
        lb = int(params["lookback"])
        thr = float(params["thr"])
        x = pd.to_numeric(df["Close"], errors="coerce").pct_change(lb)
        return pd.Series(np.where(x > thr, 1, np.where(x < -thr, -1, 0)), index=df.index)
    if family == "mean_reversion":
        zthr = float(params["zthr"])
        z = pd.to_numeric(df["Z_20"], errors="coerce")
        return pd.Series(np.where(z < -zthr, 1, np.where(z > zthr, -1, 0)), index=df.index)
    if family == "breakout":
        lb = int(params["lookback"])
        c = pd.to_numeric(df["Close"], errors="coerce")
        hh = c.rolling(lb).max().shift(1)
        ll = c.rolling(lb).min().shift(1)
        return pd.Series(np.where(c > hh, 1, np.where(c < ll, -1, 0)), index=df.index)
    if family == "regime_momentum":
        lb = int(params["lookback"])
        thr = float(params["thr"])
        mom = pd.to_numeric(df["Close"], errors="coerce").pct_change(lb)
        reg = pd.to_numeric(df.get("REGIME", 2), errors="coerce").fillna(2).astype(int)
        raw = np.where(mom > thr, 1, np.where(mom < -thr, -1, 0))
        raw = np.where(reg == 1, raw, 0)
        return pd.Series(raw, index=df.index)
    return pd.Series(0, index=df.index)


def run_edge_scanner(
    train_end: str = "2022-12-31 23:00:00",
    val_end: str = "2023-12-31 23:00:00",
    output_dir: str = "backtest/plots_oos/edge_scan",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    df = _prepare_features(df)
    df_train, df_test = _split_train_test_by_date(df, train_end=train_end, val_end=val_end)

    feature_rank = _rank_features(df_train)
    feature_rank.to_csv(out_dir / "feature_ranking.csv", index=False)
    breakout_lookbacks = _parse_breakout_lookbacks_from_env()

    families = {
        "momentum": [{"lookback": lb, "thr": thr} for lb, thr in itertools.product([4, 8, 24], [0.0, 0.0005, 0.0015])],
        "mean_reversion": [{"zthr": z} for z in [0.75, 1.0, 1.5, 2.0]],
        "breakout": [{"lookback": lb} for lb in breakout_lookbacks],
        "regime_momentum": [{"lookback": lb, "thr": thr} for lb, thr in itertools.product([4, 8, 24], [0.0, 0.0005])],
    }
    cost_cfg = {"commission": 0.0002, "spread_bps": 2.0, "slippage_bps": 1.0, "slippage_atr_coef": 0.0, "execution_delay_bars": 1}

    baseline_rows = []
    best_by_family = {}
    for fam, grid in families.items():
        best = None
        best_score = -1e18
        for params in grid:
            sig_train = _make_signal(df_train, fam, params)
            train_m = _simulate_from_signal(df_train, sig_train, **cost_cfg)
            score = float(train_m["total_return_pct"] - 0.4 * abs(min(0.0, train_m["max_drawdown_pct"])))
            if score > best_score:
                best_score = score
                best = params
        sig_train = _make_signal(df_train, fam, best)
        sig_test = _make_signal(df_test, fam, best)
        train_m = _simulate_from_signal(df_train, sig_train, **cost_cfg)
        test_m = _simulate_from_signal(df_test, sig_test, **cost_cfg)
        row = {
            "family": fam,
            "best_params": json.dumps(best, sort_keys=True),
            "train_return_pct": train_m["total_return_pct"],
            "train_dd_pct": train_m["max_drawdown_pct"],
            "train_win_rate_pct": train_m["pct_profitable"],
            "train_trades": train_m["total_trades"],
            "test_return_pct": test_m["total_return_pct"],
            "test_dd_pct": test_m["max_drawdown_pct"],
            "test_win_rate_pct": test_m["pct_profitable"],
            "test_trades": test_m["total_trades"],
            "test_expectancy": test_m["expectancy"],
        }
        baseline_rows.append(row)
        best_by_family[fam] = best

    baseline_df = pd.DataFrame(baseline_rows).sort_values("test_return_pct", ascending=False).reset_index(drop=True)
    baseline_df.to_csv(out_dir / "baseline_train_test.csv", index=False)

    # Walk-forward on top-2 families with fixed params selected on global train.
    top_families = baseline_df["family"].head(2).tolist()
    full = df.copy()
    full["Time"] = pd.to_datetime(full["Time"], errors="coerce")
    max_t = full["Time"].max()
    wf_rows = []
    for fam in top_families:
        params = best_by_family.get(fam, {})
        test_start = full["Time"].min() + pd.DateOffset(years=8)
        while test_start + pd.DateOffset(years=1) <= max_t:
            train_start = test_start - pd.DateOffset(years=8)
            train_slice = full.loc[(full["Time"] >= train_start) & (full["Time"] < test_start)].reset_index(drop=True)
            test_end = test_start + pd.DateOffset(years=1)
            test_slice = full.loc[(full["Time"] >= test_start) & (full["Time"] < test_end)].reset_index(drop=True)
            if len(train_slice) < 5000 or len(test_slice) < 1000:
                test_start += pd.DateOffset(years=1)
                continue
            sig_test = _make_signal(test_slice, fam, params)
            met = _simulate_from_signal(test_slice, sig_test, **cost_cfg)
            wf_rows.append(
                {
                    "family": fam,
                    "test_start": str(test_start.date()),
                    "test_end": str((test_end - pd.Timedelta(hours=1)).date()),
                    "test_return_pct": met["total_return_pct"],
                    "test_dd_pct": met["max_drawdown_pct"],
                    "test_win_rate_pct": met["pct_profitable"],
                    "test_trades": met["total_trades"],
                }
            )
            test_start += pd.DateOffset(years=1)

    wf_df = pd.DataFrame(wf_rows)
    if not wf_df.empty:
        wf_df.to_csv(out_dir / "walk_forward.csv", index=False)
        wf_summary = (
            wf_df.groupby("family")
            .agg(
                splits=("test_return_pct", "count"),
                median_test_return_pct=("test_return_pct", "median"),
                mean_test_return_pct=("test_return_pct", "mean"),
                positive_split_ratio=("test_return_pct", lambda x: float((x > 0).mean())),
            )
            .reset_index()
        )
    else:
        wf_summary = pd.DataFrame(columns=["family", "splits", "median_test_return_pct", "mean_test_return_pct", "positive_split_ratio"])

    summary = {
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "train_start": str(pd.to_datetime(df_train["Time"]).min()),
        "train_end": str(pd.to_datetime(df_train["Time"]).max()),
        "test_start": str(pd.to_datetime(df_test["Time"]).min()),
        "test_end": str(pd.to_datetime(df_test["Time"]).max()),
        "breakout_candidates": breakout_lookbacks,
        "top_feature_rows": feature_rank.head(10).to_dict(orient="records"),
        "baseline_rank": baseline_df.to_dict(orient="records"),
        "walk_forward_summary": wf_summary.to_dict(orient="records"),
    }
    with open(out_dir / "edge_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00").strip()
    val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00").strip()
    out = os.getenv("RL_EDGE_SCAN_DIR", "backtest/plots_oos/edge_scan").strip()
    summary = run_edge_scanner(train_end=train_end, val_end=val_end, output_dir=out)
    print("\n[EDGE SCAN] completed")
    print(f"  output_dir: {out}")
    print(f"  train rows: {summary['train_rows']}, test rows: {summary['test_rows']}")
    if summary["baseline_rank"]:
        top = summary["baseline_rank"][0]
        print(f"  best baseline on test: {top['family']} ({top['test_return_pct']:+.2f}%)")
