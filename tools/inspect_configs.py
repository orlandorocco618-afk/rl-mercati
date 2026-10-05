#!/usr/bin/env python3
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')

from backtest.out_of_sample_test import make_env
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter
from backtest.out_of_sample_test import split_train_test_by_date

# prepare data
train_end='2022-12-31 23:00:00'
val_end='2023-12-31 23:00:00'
df=merge_h1_with_d1_regimes()
df=apply_kalman_filter(df, price_col='Close')
df['Close']=df['Close_KF']
_, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)

configs = []
# main env_config
main = Path('models/env_config.json')
if main.exists():
    configs.append(('main', main))
# multi seed configs
ms = Path('models/multi_seed_runs')
if ms.exists():
    for sd in sorted(ms.iterdir()):
        if sd.is_dir():
            for stage in ['stage1_env_config.json','stage2_env_config.json']:
                p = sd / stage
                if p.exists():
                    configs.append((f'{sd.name}/{stage}', p))

for name,p in configs:
    cfg = json.load(open(p))
    env_init = make_env(df_test, allow_hold_when_flat=True, env_config=cfg)
    env = env_init()
    print(name, 'obs_space', env.observation_space, 'shape', env.observation_space.shape)
    env.close()
