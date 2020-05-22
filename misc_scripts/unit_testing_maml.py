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
from core_functions.policies import DiagNormalPolicyANIL
from core_functions.rl import fast_adapt_ppo, evaluate_ppo, set_device

#   - ML1_reach-v1, ML1_pick-place-v1, ML1_push-v1
#   - ML10, ML45
env_name = 'ML1_push-v1'
workers = 5

params = {
    # Inner loop parameters
    'ppo_epochs': 2,
    'ppo_clip_ratio': 0.1,
    'inner_lr': 0.05,
    'adapt_steps': 1,
    'adapt_batch_size': 10,  # 'shots' (will be *evenly* distributed across workers)
    # Outer loop parameters
    'meta_batch_size': 20,  # 'ways'
    'outer_lr': 0.1,
    # Common parameters
    'activation': 'tanh',  # for MetaWorld use tanh, others relu
    'tau': 1.0,
    'gamma': 0.99,
    # Other parameters
    'num_iterations': 1000,
    'save_every': 25,
    'seed': 42}


def main():
    # Set seed
    device = torch.device('cpu')
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    set_device(device)

    env = make_env(env_name, workers, params['seed'])

    task_list = env.sample_tasks(params['meta_batch_size'])
    env.set_task(task_list[0])
    env.reset()
    task = ch.envs.Runner(env)

    baseline = ch.models.robotics.LinearValue(env.state_size, env.action_size)
    policy = DiagNormalPolicyANIL(env.state_size, env.action_size, fc_neurons=100)
    policy = MAML(policy, lr=params['inner_lr'])
    meta_optimizer = torch.optim.Adam(policy.parameters(), lr=params['outer_lr'])

    test_maml_ppo_inner_loop(task, policy, baseline)

    test_maml_ppo_outer_loop(env, policy, baseline, meta_optimizer)


def test_maml_ppo_inner_loop(task, learner, baseline):
    eval_loss, task_rew = fast_adapt_ppo(task, learner, baseline, params)


def test_maml_ppo_outer_loop(env, policy, baseline, meta_optimizer):
    head_before_b = deepcopy(policy.head.weight.data.numpy())
    features_0_before_b = deepcopy(policy.body[0].weight.data.numpy())
    features_1_before_b = deepcopy(policy.body[2].weight.data.numpy())\

    meta_optimizer.zero_grad()

    iter_reward = 0.0
    iter_loss = 0.0

    task_list = env.sample_tasks(params['meta_batch_size'])

    for task_i in trange(len(task_list), leave=False, desc='Task', position=0):
        task = task_list[task_i]

        learner = policy.clone()
        env.set_task(task)
        env.reset()
        task = ch.envs.Runner(env)

        head_before = deepcopy(learner.head.weight.data.numpy())
        features_0_before = deepcopy(learner.body[0].weight.data.numpy())
        features_1_before = deepcopy(learner.body[2].weight.data.numpy())
        # Adapt
        eval_loss, task_rew = fast_adapt_ppo(task, learner, baseline, params)

        head_after = learner.head.weight.data.numpy()
        features_0_after = learner.body[0].weight.data.numpy()
        features_1_after = learner.body[2].weight.data.numpy()

        head_change = np.mean(np.abs(head_after - head_before))
        features_0_change = np.mean(np.abs(features_0_after - features_0_before))
        features_1_change = np.mean(np.abs(features_1_after - features_1_before))

        print(f"\t TASK HEAD: {head_change}")
        print(f"\t TASK FEAT 0 : {features_0_change}")
        print(f"\t TASK FEAT 1 : {features_1_change}")

        iter_reward += task_rew
        iter_loss += eval_loss
        print(f'\tTask {task_i} reward: {task_rew} | Loss : {eval_loss.item()}')

    # Log
    average_return = iter_reward / params['meta_batch_size']
    av_loss = iter_loss / params['meta_batch_size']

    print(f'Average reward: {average_return} | Loss : {av_loss}')

    # Meta-optimize: Back-propagate through the accumulated gradients and optimize
    av_loss.backward()
    meta_optimizer.step()

    # CHECK WEIGHTS AFTER OPT
    head_after_a = policy.head.weight.data.numpy()
    features_0_after_a = policy.body[0].weight.data.numpy()
    features_1_after_a = policy.body[2].weight.data.numpy()

    head_change = np.mean(np.abs(head_after_a - head_before_b))
    features_0_change = np.mean(np.abs(features_0_after_a - features_0_before_b))
    features_1_change = np.mean(np.abs(features_1_after_a - features_1_before_b))

    print(f"HEAD: {head_change}")
    print(f"FEAT 0 : {features_0_change}")
    print(f"FEAT 1 : {features_1_change}")


def fast_adapt_ppo(task, learner, baseline, params, anil=False, render=False):
    # During inner loop adaptation we do not store gradients for the network body
    if anil:
        learner.module.turn_off_body_grads()

    for step in range(params['adapt_steps']):
        # Collect adaptation / support episodes
        support_episodes = task.run(learner, episodes=params['adapt_batch_size'], render=render)

        # Calculate loss & fit the value function & update the policy
        inner_loss = ppo_update(support_episodes, learner, baseline, params, anil=anil)
        # print(f'Inner loss {step}: {round(inner_loss.item(), 3)}')

    # We need to include the body network parameters for the query set
    if anil:
        learner.module.turn_on_body_grads()

    # Collect evaluation / query episodes
    query_episodes = task.run(learner, episodes=params['adapt_batch_size'])
    # Calculate loss for the outer loop optimization WITHOUT adapting (works, tested!)
    eval_learner = learner.clone()
    eval_baseline = deepcopy(baseline)
    outer_loss = ppo_update(query_episodes, eval_learner, eval_baseline, params, anil=anil)
    # Calculate the average reward of the evaluation episodes
    query_rew = query_episodes.reward().sum().item() / params['adapt_batch_size']

    return outer_loss, query_rew


if __name__ == '__main__':
    main()


