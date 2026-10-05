import pandas as pd
from pathlib import Path

out_dir = Path('best_models')
out_dir.mkdir(exist_ok=True)

br = pd.read_csv('backtest/plots_oos/feature_grid_multi_seed/results.csv')
sr = pd.read_csv('backtest/plots_oos/feature_grid_multi_seed_sr/results.csv')

br_best = br.sort_values('total_return_pct', ascending=False).head(3)
sr_best = sr.sort_values('total_return_pct', ascending=False).head(3)

br_paths = br_best[['model_path','env_config_path']].to_dict(orient='records')
sr_paths = sr_best[['model_path','env_config_path']].to_dict(orient='records')

(Path(out_dir)/'br_top3_paths.txt').write_text('\n'.join(f"{p['model_path']} | {p['env_config_path']}" for p in br_paths))
(Path(out_dir)/'sr_top3_paths.txt').write_text('\n'.join(f"{p['model_path']} | {p['env_config_path']}" for p in sr_paths))

copy_dir = out_dir / 'copied'
copy_dir.mkdir(exist_ok=True)

for p in br_paths + sr_paths:
    for key in ['model_path','env_config_path']:
        src = Path(p[key])
        if src.exists():
            dst = copy_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())

print('Saved BR top3 paths to', out_dir / 'br_top3_paths.txt')
print('Saved SR top3 paths to', out_dir / 'sr_top3_paths.txt')
print('Copied best model files to', copy_dir)
