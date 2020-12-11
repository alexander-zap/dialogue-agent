import random
from collections import deque
from operator import itemgetter

import numpy as np


class ReplayMemory(object):
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def add(self, prev_obs, prev_act, reward, obs, done, *args):
        memory_item = (prev_obs, prev_act, reward, obs, done, *args)
        self.memory.append(memory_item)

    def sample(self, batch_size):
        pass

    def clear(self):
        self.memory.clear()


class RandomReplayMemory(ReplayMemory):
    def sample(self, batch_size):
        sample_indices = random.sample(range(self.__len__()), batch_size)
        sample_items = itemgetter(*sample_indices)(self.memory)
        return sample_items, sample_indices, np.ones(batch_size)


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, size, alpha=.9, beta=.6):
        super(PrioritizedReplayMemory, self).__init__(size)
        self._alpha = alpha
        self._beta = beta
        self._max_priority = 1.0
        self._priority_epsilon = 1e-06

    def add(self, *args):
        # Add new experience with highest existing priority
        memory_priority = self._max_priority ** self._alpha
        super().add(*args, memory_priority)

    # TODO: Implement the rank-based variant as well
    def _sample_proportional(self, batch_size, priorities):
        priorities += self._priority_epsilon
        sample_probabilities = priorities / priorities.sum()
        sampled_indices = np.random.choice(range(self.__len__()), size=batch_size, p=sample_probabilities)
        return sampled_indices

    def sample(self, batch_size):
        priorities = np.array(zip(*self.memory)[-1])
        sample_indices = self._sample_proportional(batch_size, priorities)

        sampling_correction_weights = []
        priority_min = np.amin(priorities)
        priority_sum = priorities.sum()
        p_min = priority_min / priority_sum
        max_weight = (p_min * len(self.memory)) ** (-self.beta)

        for idx in sample_indices:
            p_sample = priorities[idx] / priority_sum
            weight = (p_sample * len(self.memory)) ** (-self.beta)
            sampling_correction_weights.append(weight / max_weight)
        sampling_correction_weights = np.array(sampling_correction_weights)
        sample_items = itemgetter(*sample_indices)(self.memory)
        return sample_items, sample_indices, sampling_correction_weights

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            self.memory[idx][-1] = priority ** self._alpha
            # FIXME: _max_priority does not regulate downwards
            self._max_priority = max(self._max_priority, priority)

    # TODO: Increase beta with time
    def set_beta(self, beta):
        self._beta = beta
