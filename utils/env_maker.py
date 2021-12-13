import cherry as ch
from learn2learn.gym import AsyncVectorEnv

import gym
import utils

mujoco_envs = ['HalfCheetahPT-v2', 'HopperPT-v2', 
                'Walker2dPT-v2', 'AntPT-v2', 'SwimmerPT-v2', 'HumanoidPT-v2', 
                'HalfCheetahMetaDir-v2', 'AntDirection-v1', ]
metaworld_envs = ['ML1_reach-v1', 'ML1_pick-place-v1', 'ML1_push-v1', 'ML10', 'ML45']


def _make_mujoco(env_name, n_workers, callback=None):

    def init_env():
        env = gym.make(env_name)
        env = ch.envs.ActionSpaceScaler(env)
        if callback is not None: callback(env)
        return env

    if n_workers == 1:
        return init_env()
    else:
        return AsyncVectorEnv([init_env for _ in range(n_workers)])


def make_env(env_name, n_workers, seed, test=False, max_path_length=None, callback=None):

    if env_name in mujoco_envs:
        env = _make_mujoco(env_name, n_workers, callback=callback)
    else:
        raise NotImplementedError

    env.seed(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = ch.envs.Torch(env)
    return env


def calculate_samples_seen(n_steps, n_episodes, n_inner_steps, n_tasks, n_iters):
    n_samples = dict()
    n_samples['rollout'] = n_steps  # Samples in one episode
    n_samples['task_batch'] = n_samples['rollout'] * n_episodes  # Samples per task (steps * episodes)
    n_samples['task_support'] = n_samples['task_batch'] * n_inner_steps  # Samples adapted to per task
    # Samples in inner loop per task (support + query set)
    n_samples['task_total'] = n_samples['task_support'] + n_samples['task_batch']
    n_samples['iter'] = n_samples['task_total'] * n_tasks  # Samples in one iteration
    n_samples['total'] = n_samples['iter'] * n_iters
    return n_samples


if __name__ == '__main__':
    print(calculate_samples_seen(150, 10, 1, 20, 1000))
