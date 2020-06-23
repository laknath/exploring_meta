# Experiments on Meta Learning algorithms

Exploring the effects of Meta Learning algorithms (MAML & ANIL) on Vision and RL tasks through a different lens.

Check runs & results in [weights&biases](https://app.wandb.ai/kosz/l2l)

## Overview

#### Vision: Omniglot, Mini-ImageNet
- [x] MAML with CNN
- [x] ANIL with CNN

#### RL: Particles2D, MuJoCo Ant, Meta-World
- [X] MAML-PPO
- [X] MAML-TRPO
- [X] ANIL-PPO
- [X] ANIL-TRPO
- [X] Baselines: PPO, TRPO & VPG


## Installing

1. Install Cython:

```pip install cython```

2. Install my fork of [learn2learn](https://github.com/learnables/learn2learn) specifically modified for experiments for this repo:


```pip install git+https://github.com/Kostis-S-Z/learn2learn.git@exploring_meta#egg=learn2learn```


3. Install core dependencies

```pip install -r requirements.txt```


4. (_Optional_) Track results with [W&B](https://www.wandb.com/): `pip install wandb`


## Guide

For a vision walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/vision/VISION_CODE_WALKTHROUGH.md)

For an RL walk-through of the code check [here](https://github.com/Kostis-S-Z/exploring_meta/blob/master/rl/RL_CODE_WALKTHROUGH.md)

## Run

```
python3 rl/maml_trpo.py --outer_lr 0.1 --adapt_steps 3
```

_If you get import errors, add the project's root to PYTHONPATH. Make sure the content root is added to the PYTHONPATH in the configuration_ or _in the .bashrc file add_ `export PYTHONPATH="${PYTHONPATH}:~/Path/to/project"`


## Future extensions

_Check branches:_
- [ ] Procgen (it works, but incredibly difficult to train)
- [ ] MAML/ANIL - VPG (very unstable)

## Acknowledgements

- [learn2learn](https://github.com/learnables/learn2learn)
- [cherry-rl](https://github.com/learnables/cherry)
