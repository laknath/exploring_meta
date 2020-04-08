#!/usr/bin/env python3

import argparse
import random
import numpy as np
from tqdm import trange
from copy import deepcopy

import torch
import cherry as ch

from mpi4py import MPI
from procgen import ProcgenEnv
from baselines import logger
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import (VecExtractDictObs, VecMonitor, VecNormalize)

from utils import *
from core_functions.policies import DiagNormalPolicyCNN
from core_functions.rl import fast_adapt_a2c, meta_optimize, evaluate
from misc_scripts import run_cl_rl_exp

from sampler import Sampler

# updates = total timesteps / batch
# 1.000.000 serial timesteps takes around 3hours
# 25.000.000 timesteps for easy difficulty
# 200.000.000 timesteps for hard difficulty

params = {
    "outer_lr": 0.1,  #
    "inner_lr": 0.1,  # Default: 0.1
    "tau": 0.95,
    "gamma": 0.99,
    "backtrack_factor": 0.5,  # Meta-optimizer
    "ls_max_steps": 15,  # Meta-optimizer
    "max_kl": 0.01,  # Meta-optimizer
    "adapt_batch_size": 5,  # "shots"  Default: 20
    "meta_batch_size": 2,  # "ways" Default: 20
    "adapt_steps": 1,  # Default 1
    "num_iterations": 500,  # Default 500
    "save_every": 25,
    "seed": 42}

network = [32, 64, 64]

eval_params = {
    'n_eval_adapt_steps': 5,  # Number of steps to adapt to a new task
    'n_eval_episodes': 10,  # Number of shots per task
    'n_eval_tasks': 10,  # Number of different tasks to evaluate on
    'inner_lr': params['inner_lr'],  # Just use the default parameters for evaluating
    'tau': params['tau'],
    'gamma': params['gamma'],
}
cl_test = True
cl_params = {
    "adapt_steps": 10,
    "adapt_batch_size": 10,  # shots
    "inner_lr": 0.3,
    "gamma": 0.99,
    "tau": 1.0,
    "n_tasks": 5
}

# caveflyer, coinrun, dodgeball, maze, starpilot
env_name = "coinrun"
distribution_mode = 'easy'
num_levels = 0
start_level = 0
num_envs = 4  # 32env ~ 7gb VRAM,
test_worker_interval = 0

cuda = False

wandb = False


class MamlRL(Experiment):

    def __init__(self):
        super(MamlRL, self).__init__("maml", env_name, params, path="rl_results/", use_wandb=wandb)

        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        if cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        is_test_worker = False
        if test_worker_interval > 0:
            is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

        mpi_rank_weight = 0 if is_test_worker else 1
        n_levels = 0 if is_test_worker else num_levels

        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []

        logger.configure(dir=self.model_path, format_strs=format_strs)
        logger.info(f"Creating {num_envs} {env_name} environments")

        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=n_levels,
                          start_level=start_level, distribution_mode=distribution_mode)

        venv = VecExtractDictObs(venv, "rgb")

        venv = VecMonitor(venv=venv, filename=None, keep_buf=100, )

        venv = VecNormalize(venv=venv, ob=False)

        setup_mpi_gpus()

        self.run(venv, device)

    def run(self, env, device):

        observ_space = env.observation_space.shape[::-1]
        observ_size = len(observ_space)
        observ_space_flat = observ_space[0] * observ_space[1] * observ_space[2]
        action_space = env.action_space.n + 1

        print(observ_space)
        print(observ_size)
        print(observ_space_flat)
        print(action_space)

        baseline = ch.models.robotics.LinearValue(observ_space_flat, action_space)
        policy = DiagNormalPolicyCNN(observ_size, action_space, network=network)
        policy.to(device)

        self.log_model(policy, device, input_shape=observ_space)  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'], desc="Iteration", position=0)
        try:
            for iteration in t:

                iter_reward = 0
                iter_replays = []
                iter_policies = []

                for task_i in trange(2, leave=False, desc="Task", position=0):

                    clone = deepcopy(policy)

                    # Sampler uses policy.eval() which turns off training to sample the actions
                    sampler = Sampler(env=env, model=clone, num_steps=self.params['adapt_batch_size'],
                                      gamma_coef=params['gamma'], lambda_coef=params['tau'],
                                      device=device, num_envs=num_envs)
                    task_replay = []

                    # Adapt
                    for step in range(self.params['adapt_steps']):

                        tr_ep_samples, tr_ep_info = sampler.run()
                        task_replay.append(tr_ep_samples)
                        clone = fast_adapt_a2c(clone, tr_ep_samples, baseline,
                                               self.params['inner_lr'], self.params['gamma'], self.params['tau'],
                                               first_order=True)

                    # Compute validation Loss
                    val_ep_samples, val_ep_info = sampler.run()
                    task_replay.append(val_ep_samples)

                    iter_reward += val_ep_samples["rewards"].sum().item() / self.params['adapt_batch_size']
                    iter_replays.append(task_replay)
                    iter_policies.append(clone)

                adapt_reward = iter_reward / self.params['meta_batch_size']
                metrics = {'adapt_reward': adapt_reward}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                meta_optimize(self.params, policy, baseline, iter_replays, iter_policies, cuda)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy, str(iteration))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy)

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        self.logger['test_reward'] = evaluate(env, policy, baseline, eval_params)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()

        if cl_test:
            print("Running Continual Learning experiment...")
            run_cl_rl_exp(self.model_path, env, policy, baseline, cl_params=cl_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')

    parser.add_argument('--outer_lr', type=float, default=params['outer_lr'], help='Outer lr')
    parser.add_argument('--inner_lr', type=float, default=params['inner_lr'], help='Inner lr')
    parser.add_argument('--adapt_steps', type=int, default=params['adapt_steps'], help='Adaptation steps in inner loop')
    parser.add_argument('--meta_batch_size', type=int, default=params['meta_batch_size'], help='Batch size')
    parser.add_argument('--adapt_batch_size', type=int, default=params['adapt_batch_size'], help='Adapt batch size')

    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')

    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['outer_lr'] = args.outer_lr
    params['inner_lr'] = args.inner_lr
    params['adapt_steps'] = args.adapt_steps
    params['meta_batch_size'] = args.meta_batch_size
    params['adapt_batch_size'] = args.adapt_batch_size

    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every

    params['seed'] = args.seed

    MamlRL()
