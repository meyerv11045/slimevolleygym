"""
Multiagent example.

Evaluate the performance of different trained models in zoo against each other.

This file can be modified to test your custom models later on against existing models.

Model Choices
=============

BaselinePolicy: Default built-in opponent policy (trained in earlier 2015 project)

baseline: Baseline Policy (built-in AI). Simple 120-param RNN.
ppo: PPO trained using 96-cores for a long time vs baseline AI (train_ppo_mpi.py)
cma: CMA-ES with small network trained vs baseline AI using estool
ga: Genetic algorithm with tiny network trained using simple tournament selection and self play (input x(train_ga_selfplay.py)
random: random action agent
"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import os
import numpy as np
import argparse
import slimevolleygym
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep
from stable_baselines import PPO1

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

class PPOPolicy:
  def __init__(self, path):
    self.model = PPO1.load(path)
  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

class RandomPolicy:
  def __init__(self, path):
    self.action_space = gym.spaces.MultiBinary(3)
    pass
  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()

def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  #count = 0

  while not done:

    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)

  return total_reward

def evaluate_multiagent(env, policy0, policy1, render_mode=False, n_trials=1000, init_seed=721):
  """ Evaluates cumulative score in terms of policy0 which is the right agent
  """
  left_history = []
  right_history = []
  for i in range(n_trials):
    env.seed(seed=init_seed+i)
    cumulative_score = rollout(env, policy0, policy1, render_mode=render_mode)
    left_history.append(-1 * cumulative_score)
    print("cumulative score #", i, ":", cumulative_score)
    right_history.append(cumulative_score)

  return (left_history, right_history)

if __name__=="__main__":

  MODEL = {
    "baseline": makeBaselinePolicy,
    "ppo": PPOPolicy,
    "cma": makeSlimePolicy,
    "ga": makeSlimePolicyLite,
    "random": RandomPolicy,
  }

  parser = argparse.ArgumentParser(description='Evaluate agents against each other.')

  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)
  parser.add_argument('--day', action='store_true', help='daytime colors?', default=False)
  parser.add_argument('--pixel', action='store_true', help='pixel rendering effect? (note: not pixel obs mode)', default=False)
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  parser.add_argument('--trials', help='number of trials (default 1000)', type=int, default=1000)
  parser.add_argument('-v', '--variant', action='store_true', default=False)

  args = parser.parse_args()

  if args.day:
    slimevolleygym.setDayColors()

  if args.pixel:
    slimevolleygym.setPixelObsMode()


  if args.variant:
    env = gym.make("Variant-v0")
  else:
    env = gym.make("SlimeVolleyBaseline-v0")

  env.seed(args.seed)

  render_mode = args.render

  if args.variant:
    agents = ['ppo-sp', 'ga-sp', 'ppo']
  else:
    agents = ['a1a', 'a1b', 'a2a', 'a2b']

  def model_path(agent):
    base = "/Users/vikram/s23/ai/slimevolleygym/results"
    if agent == 'ppo-sp':
      return base + "/game_variant/ppo_selfplay/best_model.zip"
    elif agent == 'ga-sp':
      return base + "/game_variant/ga_selfplay/ga_00308000.json"
    elif agent == 'ppo':
      return base + "/game_variant/ppo367/best_model.zip"
    else:
      return f"{base}/agents/{agent}/best_model.zip"

  scores = {} # store left agent's score

  for left in agents:
    for right in agents:
      key = left + ' vs ' + right

      left_path = model_path(left)
      right_path = model_path(right)

      assert os.path.exists(left_path), left_path+" doesn't exist."
      assert os.path.exists(right_path), right_path+" doesn't exist."

      # the right agent
      if "ga" in right:
        policy0 = MODEL["ga"](right_path)
      else:
        policy0 = MODEL["ppo"](right_path)

      # the left agent
      if "ga" in left:
        policy1 = MODEL["ga"](left_path)
      else:
        policy1 = MODEL["ppo"](left_path)

      left_scores, _ = evaluate_multiagent(env, policy0, policy1,
        render_mode=render_mode, n_trials=args.trials, init_seed=args.seed)

      scores[key] = (np.round(np.mean(left_scores), 3), np.round(np.std(left_scores), 3))

  for key, val in scores.items():
    print(f"{key}: {val[0]} ± {val[1]}")

""" 100 trials
a1a vs a1b: 2.77 ± 1.561
a1a vs a2a: 0.56 ± 2.551
a1a vs a2b: 1.29 ± 2.471
a1b vs a1a: -2.87 ± 1.566
a1b vs a2a: -2.83 ± 1.898
a1b vs a2b: 0.48 ± 2.787
a2a vs a1a: -0.51 ± 2.805
a2a vs a1b: 2.78 ± 1.683
a2a vs a2b: 0.17 ± 2.657
a2b vs a1a: -0.77 ± 2.588
a2b vs a1b: 0.25 ± 2.722
a2b vs a2a: 0.12 ± 2.624

a1a vs a1a: -0.07 ± 0.587
a1b vs a1b: -0.02 ± 2.542
a2a vs a2a: 0.08 ± 2.622
a2b vs a2b: 0.13 ± 2.61
"""