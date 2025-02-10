import shutil
import gym
import torch
import torch.nn as nn
import numpy as np
import random
import os
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

# class NormalizedActions(gym.ActionWrapper):
#     def _action(self, action):
#         low = self.action_space.low
#         high = self.action_space.high
#
#         action = low + (action + 1.0) * 0.5 * (high - low)
#         action = np.clip(action, low, high)
#
#         return action


class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env):
        super(NormalizedActions, self).__init__(env)
        self.low = self.action_space.low
        self.high = self.action_space.high

    def action(self, action):
        """
        Normalize action from [-1, 1] to [low, high].
        """
        action = 0.5 * (action + 1) * (self.high - self.low) + self.low
        return np.clip(action, self.low, self.high)

    def reverse_action(self, action):
        """
        Reverse normalization from [low, high] back to [-1, 1].
        """
        action = 2 * (action - self.low) / (self.high - self.low) - 1
        return np.clip(action, -1, 1)

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    rewards = []
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        rewards = checkpoint['rewards']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, rewards


def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('episode %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()