#!/usr/bin/env python3

import argparse
import random
import torch
import numpy as np
from copy import deepcopy

from tqdm import trange, tqdm

import cherry as ch
from learn2learn.algorithms import MAML

from utils import *
from core_functions.policies import DiagNormalPolicy
from core_functions.rl import fast_adapt_vpg, evaluate_vpg
from misc_scripts import run_cl_rl_exp

params = {
    'batch_size': 20,
    'lr': 0.05,
    'dice': False,
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 25,
    'seed': 42}


# Environments:
#   - Particles2D-v1
#   - AntDirection-v1
#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45

env_name = 'Particles2D-v1'

workers = 5

wandb = False

cl_test = False
rep_test = False


class PPO(Experiment):

    def __init__(self):
        super(PPO, self).__init__('ppo', env_name, params, path='results/', use_wandb=wandb)

        # Set seed
        device = torch.device('cpu')
        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])

        env = make_env(env_name, workers, params['seed'])
        self.run(env, device)

    def run(self, env, device):

        baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
        policy = DiagNormalPolicy(env.state_size, env.action_size)

        optimizer = torch.optim.Adam(policy.parameters(), lr=self.params['lr'])

        self.log_model(policy, device, input_shape=(1, env.state_size))

        t = trange(self.params['num_iterations'], desc='Iteration', position=0)
        try:
            for iteration in t:

                iter_reward = 0.0
                iter_loss = 0.0

                task_list = env.sample_tasks(self.params['batch_size'])


                # Log
                average_return = iter_reward / self.params['batch_size']
                av_loss = iter_loss / self.params['batch_size']
                metrics = {'average_return': average_return,
                           'loss': av_loss.item()}

                t.set_postfix(metrics)
                self.log_metrics(metrics)

                if iteration % self.params['save_every'] == 0:
                    self.save_model_checkpoint(policy.module, str(iteration + 1))
                    self.save_model_checkpoint(baseline, 'baseline_' + str(iteration + 1))

        # Support safely manually interrupt training
        except KeyboardInterrupt:
            print('\nManually stopped training! Start evaluation & saving...\n')
            self.logger['manually_stopped'] = True
            self.params['num_iterations'] = iteration

        self.save_model(policy.module)
        self.save_model(baseline, name='baseline')

        self.logger['elapsed_time'] = str(round(t.format_dict['elapsed'], 2)) + ' sec'
        # Evaluate on new test tasks
        env = make_env(env_name, workers, params['seed'], test=True)
        self.log_metrics({'test_reward': self.logger['test_reward']})
        self.save_logs_to_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO on RL tasks')

    parser.add_argument('--env', type=str, default=env_name, help='Pick an environment')
    parser.add_argument('--lr', type=float, default=params['lr'], help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=params['batch_size'], help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=params['num_iterations'], help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=params['save_every'], help='Interval to save model')
    parser.add_argument('--seed', type=int, default=params['seed'], help='Seed')

    args = parser.parse_args()

    params['lr'] = args.inner_lr
    params['batch_size'] = args.meta_batch_size
    params['num_iterations'] = args.num_iterations
    params['save_every'] = args.save_every
    params['seed'] = args.seed

    PPO()
