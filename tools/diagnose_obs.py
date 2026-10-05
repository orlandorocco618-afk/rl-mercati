from importlib import import_module
import numpy as np
import traceback

print('Diagnostic: import env and utils')
try:
    from env.trading_env import TradingEnv
    from utils.data_utils import merge_h1_with_d1_regimes
except Exception as e:
    print('Import failed:', e)
    traceback.print_exc()
    raise

print('Merging H1+D1 via utils.data_utils.merge_h1_with_d1_regimes()')
try:
    df = merge_h1_with_d1_regimes()
    print('Merged df rows:', len(df))
except Exception as e:
    print('Failed to merge H1+D1:', e)
    traceback.print_exc()
    raise

init_opts_list = [
    {'df': df},
    {'df': df, 'allow_hold_when_flat': False},
    {'df': df, 'allow_hold_when_flat': True},
]

env = None
for opts in init_opts_list:
    try:
        display_opts = {k: ('DataFrame' if k=='df' else v) for k,v in opts.items()}
        print('Trying init with', display_opts)
        env = TradingEnv(**opts)
        print('Init OK with', display_opts)
        break
    except Exception as e:
        print('Init failed with', display_opts, ':', e)

if env is None:
    raise SystemExit('Could not initialize TradingEnv with any options')

try:
    obs = env.reset()
except Exception as e:
    print('Reset failed:', e)
    raise

print('Starting stepping...')
for i in range(2000):
    try:
        action = env.action_space.sample()
        step_res = env.step(action)
        # gymnasium may return (obs, reward, terminated, truncated, info)
        if isinstance(step_res, tuple):
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            elif len(step_res) == 4:
                obs, reward, done, info = step_res
            else:
                obs = step_res[0]
        else:
            obs = step_res
    except Exception as e:
        print('Step raised exception at iter', i, ':', e)
        traceback.print_exc()
        break

    try:
        arr = np.asarray(obs, dtype=float)
    except Exception as e:
        print('Could not convert obs to float array at iter', i, ':', e)
        print('Obs repr:', repr(obs)[:500])
        break

    if np.isnan(arr).any() or np.isinf(arr).any():
        print('Found NaN/Inf at step', i)
        print('Action:', action)
        print('Obs sample:', arr[:200])
        break

    if done:
        try:
            obs = env.reset()
        except Exception as e:
            print('Reset failed mid-loop:', e)
            break

else:
    print('No NaN/Inf found in', i+1, 'steps')

print('Diagnostic finished')
