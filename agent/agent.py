from abc import ABC, abstractmethod
from collections import deque


class Agent(ABC):
    def __init__(self, alpha, gamma, epsilon, epsilon_min, n_actions, n_ordinals, observation_dim, batch_size,
                 memory_len, replace_target_iter):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals
        self.input_size = observation_dim
        self.turn = 0

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_len)
        self.replace_target_iter = replace_target_iter
        self.replay_counter = 0

    @abstractmethod
    def update(self, prev_obs, prev_act, obs, reward, done):
        pass

    @abstractmethod
    def choose_action(self, obs, warm_up):
        pass

    def remember(self, prev_obs, prev_act, obs, rew, d):
        self.memory.append((prev_obs, prev_act, obs, rew, d))

    def end_episode(self, n_episodes):
        # gradually reduce epsilon after every done episode
        self.epsilon = self.epsilon - 2 / n_episodes if self.epsilon > self.epsilon_min else self.epsilon_min
