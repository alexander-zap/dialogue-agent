import random
import time
from abc import ABC, abstractmethod

from keras.models import Model

from agent.memory.memory import PrioritizedReplayMemory, UniformReplayMemory
from dialog_config import agent_rule_requests
from util_functions import index_to_agent_action, raw_agent_action_to_index


class Agent(ABC):
    def __init__(self, alpha, gamma, epsilon, epsilon_min, n_actions, n_ordinals, observation_dim, batch_size,
                 memory_len, prioritized_memory, replay_iter, replace_target_iter):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_anneal_amount = epsilon - epsilon_min
        self.n_actions = n_actions
        self.n_ordinals = n_ordinals
        self.input_size = observation_dim
        self.turn = 0

        self.batch_size = batch_size
        self.memory = PrioritizedReplayMemory(memory_len) if prioritized_memory else UniformReplayMemory(memory_len)
        self.replay_iter = replay_iter
        self.replace_target_iter = replace_target_iter
        self.replay_counter = 0

        self.eval_model = self.build_model()
        self.target_model = self.build_model()

    @abstractmethod
    def build_model(self) -> Model:
        """
        Creates a model which is able to generate predictions representing the goodness of each possible action in a
        given environment observation. Depending on the used agent, the predictions are either direct value function
        results or representations which are used to compute the value function. This model is updated iteratively in
        the course of the reinforcement learning process.

        :return model: Above described model in an initial state, which will be updated in the learning process
        """
        pass

    def update(self, prev_obs, prev_act, obs, reward, done, warm_up=False, replay=True):
        """
        Updates the current agent by remembering the last transition.
        Replays past memories regularly (every X iterations).

        :param prev_obs : Previously observed state
        :param prev_act : Action executed in the previously observed state
        :param obs : Newly observed state after executing the action
        :param reward : Newly observed reward for executing the transition
        :param done : Flag whether the episode arrived at a terminal state
        :param warm_up: Flag whether the warm up phase is currently used
        :param replay: Flag whether a replay should be done
        """
        self.remember(prev_obs, prev_act.feasible_action_index, obs, reward, done)
        if replay and not warm_up and len(self.memory) > self.batch_size:
            self.replay()

    def remember(self, prev_obs, prev_act_index, obs, rew, d):
        """
        Appends a transition to replay memory.
        All transition variables are given as parameters.

        :param prev_obs : Previously observed state
        :param prev_act_index : Index (in feasible_agent_actions) of action executed in the previously observed state
        :param obs : Newly observed state after executing the action
        :param rew : Newly observed reward for executing the transition
        :param d : Flag whether the episode arrived at a terminal state
        """
        self.memory.add(prev_obs, prev_act_index, obs, rew, d)

    @abstractmethod
    def replay(self):
        """
        Update the policy by replaying and learning from past transitions in the memory.
        """
        pass

    def choose_action(self, obs, warm_up=False):
        """
        Determines which action is chosen for the next turn given the current observation.
        Action choice can differ from greedy action policy when:
        - A random action is chosen due to epsilon greedy exploration policy
        - A predefined action sequence is used in the warm up phase

        :param obs : Current observation (state representation)
        :param warm_up : Flag whether the warm up phase is currently used

        :return action : AgentAction which should be chosen next by the agent
        """
        if warm_up:
            action = self.get_warm_up_action()
        elif random.random() < self.epsilon:
            action_index = random.randrange(self.n_actions)
            action = index_to_agent_action(action_index)
        else:
            action = self.get_greedy_action(obs)
        action.round_num = self.turn
        self.turn += 1
        return action

    def get_warm_up_action(self):
        """
        Retrieves the action which should be played by the agent in the warm up phase (based on current turn number).

        :return action : AgentAction which should be chosen next by the agent
        """
        # Agents' request sequence is defined in agent_rule_requests
        if self.turn < len(agent_rule_requests):
            raw_action = agent_rule_requests[self.turn]
        else:
            raw_action = agent_rule_requests[-1]
        feasible_action_index = raw_agent_action_to_index(raw_action)
        action = index_to_agent_action(feasible_action_index)
        return action

    @abstractmethod
    def get_greedy_action(self, obs):
        """
        Retrieves the best next action for the current observation according to the current policy.

        :param obs: Current observation (state representation)

        :return: action: AgentAction which should be chosen next by the agent according to the current policy
        """
        pass

    def empty_memory(self):
        """
        Empties replay memory (memory of past transitions).
        """
        self.memory.clear()

    def end_episode(self, n_episodes):
        """
        Agent logic which should be executed after ending an episode:
        - Gradually reduce epsilon to achieve exploitation behavior over time
        - Gradually increase beta of prioritized experience memory to anneal the sampling bias
        Currently reaching the end-value for these parameters is achieved after half the episodes (hard-coded: 0.5)
        """
        self.epsilon = self.epsilon - self.epsilon_anneal_amount / (n_episodes * 0.5) \
            if self.epsilon > self.epsilon_min else self.epsilon_min
        if isinstance(self.memory, PrioritizedReplayMemory):
            self.memory.beta = self.memory.beta + self.memory.beta_anneal_amount / (n_episodes * 0.5) \
                if self.memory.beta < 1 else self.memory.beta

    def save_agent_model(self):
        """
        Saves the value function prediction model to resources directory.
        To uniquely identify every model, the file name includes date and time.
        """
        self.eval_model.save_weights("resources/agent_models/" + time.strftime("%Y%m%d-%H%M%S") + ".h5")

    def load_agent_model(self, model_file_path):
        """
        Loads a previously saved value function prediction model to both evaluation and target model attribute.
        The model for the target model is cloned to prevent reference to same model.

        :param model_file_path: File path of previously saved (save_agent_model) model
        """
        self.target_model.load_weights(model_file_path)
        self.eval_model.load_weights(model_file_path)
