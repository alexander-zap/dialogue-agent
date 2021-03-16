import random
from collections import deque
from operator import itemgetter

import numpy as np


class ReplayMemory(object):
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def add(self, prev_obs, prev_act, reward, obs, done, priority):
        memory_item = (prev_obs, prev_act, reward, obs, done, priority)
        self.memory.append(memory_item)

    def sample(self, batch_size):
        pass

    def clear(self):
        self.memory.clear()


class UniformReplayMemory(ReplayMemory):
    def sample(self, batch_size):
        sample_indices = random.sample(range(len(self)), batch_size)
        sample_items = itemgetter(*sample_indices)(self.memory)
        return sample_items, sample_indices, np.ones(batch_size)

    def add(self, *args):
        # Add new experience with constant priority (results in uniform sampling)
        super().add(*args, 1.0)


class PrioritizedReplayMemory(ReplayMemory):
    """
    :param alpha: Amount of how much prioritization is used (0 -> uniform sampling, 1 -> greedy prioritization)
    :param beta: Amount of non-uniform sampling compensation (0 -> no compensation, 1 -> complete compensation)
    """

    def __init__(self, size, alpha=.6, beta=.4):
        super().__init__(size)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal_amount = 1.0 - beta
        self.priority_epsilon = 1e-03
        self.max_priority = 1.0
        self.min_priority = self.max_priority

    def add(self, *args):
        # Add new experience with highest existing priority
        super().add(*args, self.max_priority)

    def sample(self, batch_size):
        def compute_probabilities(rank_based=False):
            # Rank-based prioritization
            if rank_based:
                # FIXME
                rank_probabilities = [1/n for n in range(1, len(self))]
                return rank_probabilities
            # Proportional prioritization
            else:
                return priorities / priorities.sum()

        def compute_correction_weights():
            def compute_sampling_correction_weight(idx):
                p_sample = probabilities[idx]
                weight = (p_sample * n_memory) ** (-self.beta)
                return weight

            n_memory = len(self)
            correction_weights = np.array(list(map(compute_sampling_correction_weight, sampled_indices)))
            # Normalize sampling correction weights to [0;1]
            max_correction_weight = (self.min_priority * n_memory) ** (-self.beta)
            correction_weights = correction_weights / max_correction_weight
            return correction_weights

        def sample_indices():
            indices = np.random.choice(range(len(self)), size=batch_size, p=probabilities)
            return indices

        priorities = np.array(list(zip(*self.memory))[-1])
        probabilities = compute_probabilities()
        sampled_indices = sample_indices()
        sampling_correction_weights = compute_correction_weights()
        sampled_items = itemgetter(*sampled_indices)(self.memory)
        return sampled_items, sampled_indices, sampling_correction_weights

    def update_priorities(self, idxes, td_errors):
        for idx, td_error in zip(idxes, td_errors):
            memory_entry = list(self.memory[idx])
            # Pre-compute alpha power-operation for updated priorities
            memory_entry[-1] = (td_error + self.priority_epsilon) ** self.alpha
            self.memory[idx] = tuple(memory_entry)

        priorities = np.array(list(zip(*self.memory))[-1])
        self.min_priority = np.amin(priorities)
        self.max_priority = np.amax(priorities)
