#!/usr/bin/env python3

import learn2learn as l2l
import torch
from scipy.stats import truncnorm

from utils.data_pre import prepare_batch


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, features=None):
    adapt_data, adapt_labels, eval_data, eval_labels = prepare_batch(batch, shots, ways, device, features=features)

    for step in range(adaptation_steps):
        # Calculate loss based on predictions & actual labels
        train_loss = loss(learner(adapt_data), adapt_labels)
        # Calculate gradients & update the model's parameters based on the loss
        learner.adapt(train_loss)

    predictions = learner(eval_data)
    valid_loss = loss(predictions, eval_labels)
    valid_accuracy = accuracy(predictions, eval_labels)
    return valid_loss, valid_accuracy


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def evaluate(params, test_tasks, model, loss, device, features=None):
    meta_test_loss = 0.0
    meta_test_accuracy = 0.0
    for task in range(params['meta_batch_size']):
        # Compute meta-testing loss
        learner = model.clone()
        batch = test_tasks.sample()

        eval_loss, eval_acc = fast_adapt(batch, learner, loss,
                                         params['adapt_steps'], params['shots'], params['ways'],
                                         device, features=features)
        meta_test_loss += eval_loss.item()
        meta_test_accuracy += eval_acc.item()

    meta_test_accuracy = meta_test_accuracy / params['meta_batch_size']
    print('Meta Test Accuracy', meta_test_accuracy)
    return meta_test_accuracy


""" Code below is from learn2learn repository with slight modifications to return the representation of the layers."""


class OmniglotCNN(torch.nn.Module):
    """

    [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models.py)

    **Description**

    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

    This network assumes inputs of shapes (1, 28, 28).

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~

    """

    def __init__(self, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=1,
                             max_pool=False,
                             layers=layers)
        self.features = torch.nn.Sequential(
            l2l.nn.Lambda(lambda x: x.view(-1, 1, 28, 28)),
            self.base,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_base_representation(self, x):
        return self.base(x)


class MiniImagenetCNN(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=32) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = MiniImagenetCNN(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=32, layers=4):
        super(MiniImagenetCNN, self).__init__()
        base = ConvBase(
            output_size=hidden_size,
            hidden=hidden_size,
            channels=3,
            max_pool=True,
            layers=layers,
            max_pool_factor=4 // layers,
        )
        self.features = torch.nn.Sequential(
            base,
            l2l.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(
            25 * hidden_size,
            output_size,
            bias=True,
        )
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_base_representation(self, x):
        return self.base(x)

    def get_rep_layer(self, x, layer):
        return torch.nn.Sequential(*list(self.base.children())[:layer])(x)


class ConvBase(torch.nn.Sequential):

    # NOTE:
    #     Omniglot: hidden=64, channels=1, no max_pool
    #     MiniImagenet: hidden=32, channels=3, max_pool

    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor),
                ]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class ConvBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(
            out_channels,
            affine=True,
            # eps=1e-3,
            # momentum=0.999,
            # track_running_stats=False,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor
