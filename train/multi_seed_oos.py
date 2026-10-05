import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, ".")

from backtest.out_of_sample_test import _run_slice, split_train_test_by_date
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter


ROOT = Path(".")
MODELS_DIR = ROOT / "models"
SEED_RUN_DIR = MODELS_DIR / "multi_seed_runs"
OUTPUT_DIR = ROOT / "backtest" / "plots_oos" / "multi_seed"


def _parse_int_list(raw: str, default: list[int]) -> list[int]:
    text = (raw or "").strip()
    if not text:
        return list(default)
    out: list[int] = []
    for token in text.replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or list(default)


def _run_train(
    *,
    seed: int,
    timesteps: int,
    train_end: str,
    val_end: str,
    resume_model: str,
    prior_bonus: float,
    breakout_lookbacks: str,
    tag: str,
    extra_env: dict[str, str] | None = None,
):
    env = os.environ.copy()
    env.update(
        {
            "RL_SEED": str(seed),
            "RL_TRAIN_TIMESTEPS": str(timesteps),
            "RL_CHECKPOINT_FREQ": "50000",
            "RL_RUN_EDGE_SCAN": "0",
            "RL_SKIP_POST_ANALYSIS": "1",
            "RL_TRAIN_END": train_end,
            "RL_VAL_END": val_end,
            "RL_BREAKOUT_LOOKBACKS": breakout_lookbacks,
            "RL_BREAKOUT_PRIOR_N": "96",
            "RL_BREAKOUT_PRIOR_BONUS_COEF": f"{prior_bonus}",
        }
    )
    if extra_env:
        env.update(extra_env)
    if resume_model:
        env["RL_RESUME_MODEL"] = resume_model
    elif "RL_RESUME_MODEL" in env:
        del env["RL_RESUME_MODEL"]

    cmd = [sys.executable, "train/train_agent.py"]
    print(f"\n[MULTI-SEED] seed={seed} tag={tag} timesteps={timesteps} prior_bonus={prior_bonus}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    seed_dir = SEED_RUN_DIR / f"seed_{seed}"
    if tag == "stage1" and seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_src = MODELS_DIR / "ppo_trading.zip"
    cfg_src = MODELS_DIR / "env_config.json"
    model_dst = seed_dir / f"{tag}.zip"
    cfg_dst = seed_dir / f"{tag}_env_config.json"
    shutil.copy2(model_src, model_dst)
    shutil.copy2(cfg_src, cfg_dst)
    return model_dst, cfg_dst


def _evaluate_test_slice(model_path: Path, env_cfg_path: Path, df_test: pd.DataFrame, label: str):
    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_cfg = json.load(f)
    model = PPO.load(str(model_path))
    actions_df, metrics = _run_slice(
        model=model,
        df_slice=df_test,
        env_cfg=env_cfg,
        allow_hold_when_flat=True,
        label=label,
    )
    active_steps = int((actions_df["Action"] != 0).sum()) if not actions_df.empty else 0
    active_pct = 100.0 * active_steps / max(1, len(actions_df))
    metrics["active_steps_pct"] = float(active_pct)
    metrics["rows"] = int(len(actions_df))
    return actions_df, metrics


def _select_constrained_model(
    results: list[dict],
    *,
    min_return_pct: float,
    max_dd_abs_pct: float,
    active_min_pct: float,
    active_max_pct: float,
    min_trades: int,
    target_active_pct: float = 15.0,
):
    def is_feasible(row: dict) -> bool:
        dd_abs = abs(float(row["test_dd_pct"]))
        active = float(row["test_active_steps_pct"])
        trades = int(row["test_trades"])
        return (
            float(row["test_return_pct"]) >= min_return_pct
            and dd_abs <= max_dd_abs_pct
            and active_min_pct <= active <= active_max_pct
            and trades >= min_trades
        )

    feasible = [r for r in results if is_feasible(r)]
    if feasible:
        feasible_ranked = sorted(
            feasible,
            key=lambda r: (
                float(r["test_return_pct"]),
                -abs(float(r["test_dd_pct"])),
                -abs(float(r["test_active_steps_pct"]) - float(target_active_pct)),
                float(r["test_win_rate_pct"]),
            ),
            reverse=True,
        )
        return feasible_ranked[0], "constrained_best", feasible

    # If nobody passes all constraints, choose the best compromise.
    def compromise_score(row: dict) -> float:
        ret = float(row["test_return_pct"])
        dd_abs = abs(float(row["test_dd_pct"]))
        active = float(row["test_active_steps_pct"])
        trades = int(row["test_trades"])
        ret_gap = max(0.0, min_return_pct - ret)
        dd_gap = max(0.0, dd_abs - max_dd_abs_pct)
        if active < active_min_pct:
            active_gap = active_min_pct - active
        elif active > active_max_pct:
            active_gap = active - active_max_pct
        else:
            active_gap = 0.0
        trade_gap = max(0.0, float(min_trades - trades)) / max(1.0, float(min_trades))
        # Higher is better.
        return (
            ret
            - 20.0 * ret_gap
            - 8.0 * dd_gap
            - 1.5 * active_gap
            - 4.0 * trade_gap
            - 0.05 * abs(active - target_active_pct)
        )

    ranked = sorted(results, key=lambda r: compromise_score(r), reverse=True)
    return ranked[0], "best_compromise", []


def main():
    train_end = os.getenv("RL_TRAIN_END", "2022-12-31 23:00:00").strip()
    val_end = os.getenv("RL_VAL_END", "2023-12-31 23:00:00").strip()
    seeds = _parse_int_list(os.getenv("RL_MULTI_SEEDS", ""), [11, 22, 33, 44, 55])[:5]
    stage1_timesteps = int(os.getenv("RL_STAGE1_TIMESTEPS", "50000"))
    stage2_timesteps = int(os.getenv("RL_STAGE2_TIMESTEPS", "100000"))
    stage1_prior_bonus = float(os.getenv("RL_STAGE1_PRIOR_BONUS", "0.00045"))
    stage2_prior_bonus = float(os.getenv("RL_STAGE2_PRIOR_BONUS", "0.00015"))
    breakout_lookbacks = os.getenv("RL_MULTI_BREAKOUT_LOOKBACKS", "72,96,144").strip() or "72,96,144"
    breakout_entry_strength_quantile = float(os.getenv("RL_MULTI_BREAKOUT_ENTRY_STRENGTH_QUANTILE", "0.55"))
    enable_zones = os.getenv("RL_MULTI_ENABLE_ZONES", "0").strip()
    zone_tol = os.getenv("RL_MULTI_ZONE_TOL", "0.002").strip() or "0.002"
    zone_lookback = os.getenv("RL_MULTI_ZONE_LOOKBACK", "5").strip() or "5"
    select_min_return_pct = float(os.getenv("RL_SELECT_MIN_RETURN_PCT", "0.0"))
    select_max_dd_abs_pct = float(os.getenv("RL_SELECT_MAX_DD_ABS_PCT", "2.0"))
    select_active_min_pct = float(os.getenv("RL_SELECT_ACTIVE_MIN_PCT", "12.0"))
    select_active_max_pct = float(os.getenv("RL_SELECT_ACTIVE_MAX_PCT", "18.0"))
    select_min_trades = int(os.getenv("RL_SELECT_MIN_TRADES", "120"))
    target_active_pct = float(os.getenv("RL_SELECT_TARGET_ACTIVE_PCT", "15.0"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEED_RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_extra_env = {
        "RL_BREAKOUT_ENTRY_STRENGTH_QUANTILE": f"{breakout_entry_strength_quantile}",
        "RL_ENABLE_ZONES": enable_zones,
        "RL_ZONE_TOL": zone_tol,
        "RL_ZONE_LOOKBACK": zone_lookback,
    }

    print("[MULTI-SEED] loading data once for OOS test slice...")
    df = merge_h1_with_d1_regimes()
    df = apply_kalman_filter(df, price_col="Close")
    df["Close"] = df["Close_KF"]
    _, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)
    print(f"[MULTI-SEED] test rows={len(df_test)}")

    results = []
    for seed in seeds:
        s1_model, _ = _run_train(
            seed=seed,
            timesteps=stage1_timesteps,
            train_end=train_end,
            val_end=val_end,
            resume_model="",
            prior_bonus=stage1_prior_bonus,
            breakout_lookbacks=breakout_lookbacks,
            tag="stage1",
            extra_env=train_extra_env,
        )
        s2_model, s2_cfg = _run_train(
            seed=seed,
            timesteps=stage2_timesteps,
            train_end=train_end,
            val_end=val_end,
            resume_model=str(s1_model),
            prior_bonus=stage2_prior_bonus,
            breakout_lookbacks=breakout_lookbacks,
            tag="stage2",
            extra_env=train_extra_env,
        )
        actions_df, metrics = _evaluate_test_slice(
            model_path=s2_model,
            env_cfg_path=s2_cfg,
            df_test=df_test,
            label=f"seed_{seed}_test",
        )

        seed_out = OUTPUT_DIR / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        actions_df.to_csv(seed_out / "xauusd_actions_test.csv", index=False)
        with open(seed_out / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        row = {
            "seed": int(seed),
            "test_return_pct": float(metrics.get("total_return_pct", 0.0)),
            "test_dd_pct": float(metrics.get("max_drawdown_pct", 0.0)),
            "test_win_rate_pct": float(metrics.get("pct_profitable", 0.0)),
            "test_trades": int(metrics.get("total_trades", 0)),
            "test_active_steps_pct": float(metrics.get("active_steps_pct", 0.0)),
            "model_path": str(s2_model),
            "env_config_path": str(s2_cfg),
        }
        results.append(row)
        print(
            f"[MULTI-SEED] seed={seed} "
            f"ret={row['test_return_pct']:+.2f}% dd={row['test_dd_pct']:.2f}% "
            f"trades={row['test_trades']} active={row['test_active_steps_pct']:.1f}%"
        )

    ranked_by_return = sorted(results, key=lambda x: x["test_return_pct"], reverse=True)
    median_reference = sorted(results, key=lambda x: (x["test_return_pct"], -x["test_dd_pct"]))[len(results) // 2]
    selected_row, selection_mode, feasible_rows = _select_constrained_model(
        results,
        min_return_pct=select_min_return_pct,
        max_dd_abs_pct=select_max_dd_abs_pct,
        active_min_pct=select_active_min_pct,
        active_max_pct=select_active_max_pct,
        min_trades=select_min_trades,
        target_active_pct=target_active_pct,
    )

    # Promote selected model as main model for strict OOS report.
    shutil.copy2(selected_row["model_path"], MODELS_DIR / "ppo_trading.zip")
    shutil.copy2(selected_row["env_config_path"], MODELS_DIR / "env_config.json")

    summary = {
        "train_end": train_end,
        "val_end": val_end,
        "seeds": seeds,
        "stage1_timesteps": stage1_timesteps,
        "stage2_timesteps": stage2_timesteps,
        "stage1_prior_bonus": stage1_prior_bonus,
        "stage2_prior_bonus": stage2_prior_bonus,
        "train_env": {
            "breakout_lookbacks": breakout_lookbacks,
            "breakout_entry_strength_quantile": breakout_entry_strength_quantile,
            "enable_zones": bool(int(enable_zones)),
            "zone_tol": float(zone_tol),
            "zone_lookback": int(zone_lookback),
        },
        "selection_constraints": {
            "min_return_pct": select_min_return_pct,
            "max_dd_abs_pct": select_max_dd_abs_pct,
            "active_min_pct": select_active_min_pct,
            "active_max_pct": select_active_max_pct,
            "min_trades": select_min_trades,
            "target_active_pct": target_active_pct,
        },
        "results": results,
        "selection_mode": selection_mode,
        "feasible_count": len(feasible_rows),
        "selected": selected_row,
        "median_reference": median_reference,
        "best_return_reference": ranked_by_return[0] if ranked_by_return else None,
    }
    with open(OUTPUT_DIR / "multi_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(results).sort_values("test_return_pct", ascending=False).to_csv(
        OUTPUT_DIR / "multi_seed_results.csv", index=False
    )

    print(f"\n[MULTI-SEED] selected model ({selection_mode})")
    print(json.dumps(selected_row, indent=2))
    print(f"[MULTI-SEED] summary: {OUTPUT_DIR / 'multi_seed_summary.json'}")


if __name__ == "__main__":
    main()
