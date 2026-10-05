import json
from stable_baselines3 import PPO
from backtest.out_of_sample_test import make_env, split_train_test_by_date
from utils.data_utils import merge_h1_with_d1_regimes
from utils.kalman import apply_kalman_filter

# load model
model = PPO.load('models/ppo_trading.zip')
print('model obs_space:', model.observation_space)

# choose config path
cfg_path = Path('models/multi_seed_runs/seed_33/stage2_env_config.json')
if not cfg_path.exists():
    cfg_path = Path('models/env_config.json')
print('using config', cfg_path)
env_cfg = json.load(open(cfg_path))

# prepare data
train_end='2022-12-31 23:00:00'
val_end='2023-12-31 23:00:00'
df=merge_h1_with_d1_regimes()
df=apply_kalman_filter(df, price_col='Close')
df['Close']=df['Close_KF']
_, df_test = split_train_test_by_date(df, train_end=train_end, val_end=val_end)

# build env and inspect
env_init = make_env(df_test, allow_hold_when_flat=True, env_config=env_cfg)
env = env_init()
print('env obs_space', env.observation_space, env.observation_space.shape)
