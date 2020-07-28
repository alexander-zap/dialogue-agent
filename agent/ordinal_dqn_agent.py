from agent.agent import Agent
from util_functions import index_to_agent_action, raw_agent_action_to_index
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dialog_config import agent_rule_requests


class OrdinalDQNAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_action_nets = [self.build_model() for _ in range(self.n_actions)]
        self.target_action_nets = [self.build_model() for _ in range(self.n_actions)]

    # Creates neural net for DQN
    def build_model(self):
        model = Sequential()
        model.add(Dense(80, input_dim=self.input_size, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.n_ordinals, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def update(self, prev_obs, prev_act, obs, reward, done, warm_up=False, replay=True):
        ordinal = self.reward_to_ordinal(reward)
        self.remember(prev_obs, prev_act.feasible_action_index, obs, ordinal, done)
        if replay and not warm_up and len(self.memory) > self.batch_size:
            self.replay()

    def remember(self, prev_obs, prev_act_index, obs, ordinal, d):
        self.memory.append((prev_obs, prev_act_index, obs, ordinal, d))

    def replay(self):
        # copy evaluation model to target model at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            for a in range(self.n_actions):
                self.target_action_nets[a].set_weights(self.eval_action_nets[a].get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [[] for _ in range(self.n_actions)], [[] for _ in range(self.n_actions)]
        obs_batch = np.array([sample[2][0] for sample in mini_batch])
        obs_prediction_batch = [target_model.predict(obs_batch) for target_model in self.target_action_nets]
        borda_scores_batch = self.compute_borda_scores(obs_batch)
        for i, (prev_obs, prev_act, obs, ordinal, d) in enumerate(mini_batch):
            if not d:
                best_act = np.argmax(borda_scores_batch[i])
                target = self.gamma * obs_prediction_batch[best_act][i]
                target[ordinal] += 1
            else:
                target = np.zeros(self.n_ordinals)
                target[ordinal] += 1
            # fit predicted value of previous action in previous observation to target value of max_action
            x_batch[prev_act].append(prev_obs[0])
            y_batch[prev_act].append(target)
        for a in range(self.n_actions):
            if len(x_batch[a]) != 0:
                self.eval_action_nets[a].fit(np.array(x_batch[a]), np.array(y_batch[a]),
                                             batch_size=self.batch_size, verbose=0)

    # Chooses action with epsilon greedy exploration policy
    def choose_action(self, obs, warm_up=False):
        # Choose random action with probability epsilon
        if warm_up:
            action = self.get_warm_up_action()
        # Greedy or warm up action is chosen with probability (1 - epsilon)
        elif random.random() < self.epsilon:
            action_index = random.randrange(self.n_actions)
            action = index_to_agent_action(action_index)
        else:
            action = self.get_greedy_action(obs)
        action.round_num = self.turn
        self.turn += 1
        return action

    # Computes borda_values for a batch of observations given the ordinal_values
    def compute_borda_scores(self, obs_batch):
        obs_prediction_batch = [eval_model.predict(obs_batch) for eval_model in self.eval_action_nets]
        borda_scores_batch = []
        for i_sample in range(len(obs_batch)):
            # sum up all ordinal values per action for given observation
            ordinal_value_sum_per_action = np.zeros(self.n_actions)
            ordinal_values_per_action = [[] for _ in range(self.n_actions)]
            for action_a in range(self.n_actions):
                for ordinal_value in obs_prediction_batch[action_a][i_sample]:
                    ordinal_value_sum_per_action[action_a] += ordinal_value
                    ordinal_values_per_action[action_a].append(ordinal_value)
            ordinal_values_per_action = np.array(ordinal_values_per_action)

            # count actions whose ordinal value sum is not zero (no comparision possible for actions without ordinal_value)
            non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
            actions_to_compare_count = non_zero_action_count - 1

            borda_scores = []
            # compute borda_values for action_a (probability that action_a wins against any other action)
            for action_a in range(self.n_actions):
                # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
                if ordinal_value_sum_per_action[action_a] == 0:
                    borda_scores.append(1.0)
                    continue

                if actions_to_compare_count < 1:
                    # set lower than 1.0 (borda_value for zero_actions is 1.0)
                    borda_scores.append(0.5)
                else:
                    # over all actions: sum up the probabilities that action_a wins against the given action
                    winning_probability_a_sum = 0
                    # compare action_a to all other actions
                    for action_b in range(self.n_actions):
                        if action_a == action_b:
                            continue
                        # not comparable if action_b has no ordinal_values
                        if ordinal_value_sum_per_action[action_b] == 0:
                            continue
                        else:
                            # probability that action_a wins against action_b
                            winning_probability_a = 0
                            # running ordinal probability that action_b is worse than current investigated ordinal
                            worse_probability_b = 0
                            # predict ordinal values for action a and b
                            ordinal_values_a = ordinal_values_per_action[action_a]
                            ordinal_values_b = ordinal_values_per_action[action_b]
                            for ordinal_count in range(self.n_ordinals):
                                ordinal_probability_a = ordinal_values_a[ordinal_count] \
                                                        / ordinal_value_sum_per_action[action_a]
                                # ordinal_probability_b is also the tie probability
                                ordinal_probability_b = (ordinal_values_b[ordinal_count] /
                                                         ordinal_value_sum_per_action[action_b])
                                winning_probability_a += ordinal_probability_a * \
                                                         (worse_probability_b + ordinal_probability_b / 2.0)
                                worse_probability_b += ordinal_probability_b
                            winning_probability_a_sum += winning_probability_a
                    # normalize summed up probabilities with number of actions that have been compared
                    borda_scores.append(winning_probability_a_sum / actions_to_compare_count)
            borda_scores_batch.append(borda_scores)
        return borda_scores_batch

    def get_greedy_action(self, obs):
        action_index = np.argmax(self.compute_borda_scores([obs])[0])
        return index_to_agent_action(action_index)

    def get_warm_up_action(self):
        # Agents' request rules are defined in dialogue_config.py
        if self.turn < len(agent_rule_requests):
            raw_action = agent_rule_requests[self.turn]
            feasible_action_index = raw_agent_action_to_index(raw_action)
            return index_to_agent_action(feasible_action_index)
        else:
            raw_action = agent_rule_requests[-1]
            feasible_action_index = raw_agent_action_to_index(raw_action)
            return index_to_agent_action(feasible_action_index)

    def empty_memory(self):
        self.memory.clear()

    @staticmethod
    def reward_to_ordinal(reward):
        if reward < -1:
            return 0
        elif reward == -1:
            return 1
        elif reward > -1:
            return 2
