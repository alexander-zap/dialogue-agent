from agent.agent import Agent
from util_functions import index_to_agent_action, raw_agent_action_to_index
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from dialog_config import agent_rule_requests


class OrdinalDQNAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_model = self.build_model()
        self.target_model = self.build_model(target=True)

    # Creates big neural net for DQN
    def build_model(self, target=False):
        input_layer = Input(shape=(self.input_size,))
        hidden_layer_1 = Dense(80, activation='relu')(input_layer)
        hidden_layer_2 = Dense(20, activation='relu')(hidden_layer_1)
        action_net_outputs = [self.build_action_net(hidden_layer_2) for _ in range(self.n_actions)]
        model = Model(inputs=input_layer, outputs=action_net_outputs)
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        model.summary()
        return model

    # Creates subnet for each action
    def build_action_net(self, action_net_input):
        action_net_output = Dense(self.n_ordinals, activation='linear')(action_net_input)
        return action_net_output

    def predict(self, obs_batch, target=False):
        if target:
            return self.target_model(obs_batch)
        else:
            return self.eval_model(obs_batch)

    def replay(self):
        # copy evaluation model to target model at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [], [[] for _ in range(self.n_actions)]
        obs_batch = np.array([sample[2] for sample in mini_batch])
        obs_eval_prediction_batch = np.array(self.predict(obs_batch))
        obs_target_prediction_batch = np.array(self.predict(obs_batch, target=True))
        borda_scores_batch = self.compute_measure_of_statistical_superiority(obs_batch)
        for i, (prev_obs, prev_act, obs, reward, d) in enumerate(mini_batch):
            ordinal = self.reward_to_ordinal(reward)
            obs_eval_prediction = obs_eval_prediction_batch[:, i]
            obs_target_prediction = obs_target_prediction_batch[:, i]
            if not d:
                best_act = np.argmax(borda_scores_batch[i])
                ordinal_q_distribution = self.gamma * np.array(obs_target_prediction[best_act])
                ordinal_q_distribution[ordinal] += 1
            else:
                ordinal_q_distribution = np.zeros(self.n_ordinals)
                ordinal_q_distribution[ordinal] += 1
            # fit predicted value of previous action in previous observation to target value of max_action
            target = obs_eval_prediction
            target[prev_act] = ordinal_q_distribution

            x_batch.append(prev_obs)
            for act_idx in range(self.n_actions):
                y_batch[act_idx].append(target[act_idx])

        if len(x_batch) != 0:
            x_batch = np.array(x_batch)
            y_batch = [np.asarray(y) for y in y_batch]
            self.eval_model.fit(x_batch, y_batch, batch_size=64, verbose=1)

    # Computes the Borda counts for a batch of observations given the ordinal_values
    def compute_borda_count(self, obs_batch):
        obs_prediction_batch = self.predict(np.array(obs_batch))
        borda_counts_batch = []
        for i_sample in range(len(obs_batch)):
            # sum up all ordinal values per action for given observation
            ordinal_value_sum_per_action = np.zeros(self.n_actions)
            ordinal_values_per_action = [[] for _ in range(self.n_actions)]
            for action_a in range(self.n_actions):
                for ordinal_value in obs_prediction_batch[action_a][i_sample]:
                    ordinal_value_sum_per_action[action_a] += ordinal_value
                    ordinal_values_per_action[action_a].append(ordinal_value)
            ordinal_values_per_action = np.array(ordinal_values_per_action)

            borda_counts = []
            for action_a in range(self.n_actions):
                action_score = 0
                ordinal_values = ordinal_values_per_action[action_a]
                ordinal_worth = 1.0
                for ordinal_value in reversed(ordinal_values):
                    ordinal_probability = ordinal_value / ordinal_value_sum_per_action[action_a]
                    action_score += ordinal_probability * ordinal_worth
                    ordinal_worth = ordinal_worth / 2
                borda_counts.append(action_score)
            borda_counts_batch.append(borda_counts)
        return borda_counts_batch

    # Computes the winning probabilities of actions for a batch of observations given the ordinal_values
    def compute_measure_of_statistical_superiority(self, obs_batch):
        obs_prediction_batch = self.predict(np.array(obs_batch))
        winning_probabilities_batch = []
        for i_sample in range(len(obs_batch)):
            # sum up all ordinal values per action for given observation
            ordinal_value_sum_per_action = np.zeros(self.n_actions)
            ordinal_values_per_action = [[] for _ in range(self.n_actions)]
            for action_a in range(self.n_actions):
                for ordinal_value in obs_prediction_batch[action_a][i_sample]:
                    ordinal_value_sum_per_action[action_a] += ordinal_value
                    ordinal_values_per_action[action_a].append(ordinal_value)
            ordinal_values_per_action = np.array(ordinal_values_per_action)

            # count actions whose ordinal value sum is not zero
            # (no comparision possible for actions without ordinal_value)
            non_zero_action_count = np.count_nonzero(ordinal_value_sum_per_action)
            actions_to_compare_count = non_zero_action_count - 1

            # compute winning probabilities per action
            winning_probabilities = []
            # compute probability for every action_a that action_a wins against any other action
            for action_a in range(self.n_actions):
                # if action has not yet recorded any ordinal values, action has to be played (set borda_value to 1.0)
                if ordinal_value_sum_per_action[action_a] == 0:
                    winning_probabilities.append(1.0)
                    continue

                if actions_to_compare_count < 1:
                    # set lower than 1.0 (borda_value for zero_actions is 1.0)
                    winning_probabilities.append(0.5)
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
                    winning_probabilities.append(winning_probability_a_sum / actions_to_compare_count)
            winning_probabilities_batch.append(winning_probabilities)
        return winning_probabilities_batch

    def get_greedy_action(self, obs):
        action_index = np.argmax(self.compute_measure_of_statistical_superiority([obs])[0])
        return index_to_agent_action(action_index)

    @staticmethod
    def reward_to_ordinal(reward):
        if reward < -1:
            return 0
        elif reward == -1:
            return 1
        elif reward > -1:
            return 2
