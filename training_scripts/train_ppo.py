#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--seed', type=int, default=367)
ap.add_argument('-e', '--env', type=str, default="baseline", help="[baseline, ppo, variant]")
ap.add_argument('-g', '--gamma', type=float, default=0.99)
ap.add_argument('--entcoeff', type=float, default=0.0)
ap.add_argument('--subfolder', type=str,required=True)
args = ap.parse_args()

NUM_TIMESTEPS = int(2e7)
SEED = args.seed
EVAL_FREQ = 250000
EVAL_EPISODES = 1000

if args.env == 'baseline':
    LOGDIR = f"results/{args.subfolder}/baseline{SEED}"
else:
    LOGDIR = f"results/{args.subfolder}/ppo{SEED}"

logger.configure(folder=LOGDIR)

if args.env == 'ppo':
    env = gym.make("SlimeVolleyPPOExpert-v0")
elif args.env == 'variant':
    env = gym.make("Variant-v0")
elif args.env == 'baseline':
    env = gym.make("SlimeVolleyBaseline-v0")
else:
    raise ValueError("env not supported")

env.seed(SEED)

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=args.entcoeff, optim_epochs=10,
                 optim_stepsize=3e-4, optim_batchsize=64, gamma=args.gamma, lam=0.95, schedule='linear', verbose=2)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
