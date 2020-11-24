from agent.agent import Agent
from util_functions import index_to_agent_action
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam


class DQNAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        """
        Creates a neural network in order to predict Q-values per action given an observation (Deep Q-Network)
        """
        input_layer = Input(shape=(self.input_size,))
        hidden_layer_1 = Dense(20, activation='relu')(input_layer)
        action_net_outputs = [self.build_action_net(hidden_layer_1) for _ in range(self.n_actions)]
        model = Model(inputs=input_layer, outputs=action_net_outputs)
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        model.summary()
        return model

    # Creates subnet for each action
    @staticmethod
    def build_action_net(action_net_input):
        action_net_hidden_layer_1 = Dense(8, activation='relu')(action_net_input)
        action_net_output = Dense(1, activation='linear')(action_net_hidden_layer_1)
        return action_net_output

    def predict(self, obs_batch, target=False):
        batch_size = len(obs_batch)
        if target:
            return self.target_model.predict(obs_batch, batch_size=batch_size)
        else:
            return self.eval_model.predict(obs_batch, batch_size=batch_size)

    def replay(self):
        """
        Update the neural network (policy) by replaying and learning from a mini batch of past transitions in memory.
        The loss is computed by the difference of observed and expected network output.
        The latter is computed by the Bellman equation.
        """
        # Copy eval to target model at first replay and regularly afterwards (frequency defined in replace_target_iter)
        if self.replay_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [], [[] for _ in range(self.n_actions)]
        prev_obs_batch = np.array([sample[0] for sample in mini_batch])
        obs_batch = np.array([sample[2] for sample in mini_batch])
        prev_obs_eval_prediction_batch = np.array(self.predict(prev_obs_batch))
        obs_eval_prediction_batch = np.array(self.predict(obs_batch))
        obs_target_prediction_batch = np.array(self.predict(obs_batch, target=True))
        for i, (prev_obs, prev_act, obs, reward, d) in enumerate(mini_batch):
            prev_obs_eval_prediction = prev_obs_eval_prediction_batch[:, i]
            obs_eval_prediction = obs_eval_prediction_batch[:, i]
            obs_target_prediction = obs_target_prediction_batch[:, i]
            if not d:
                best_act = np.argmax(obs_eval_prediction)
                target = reward + self.gamma * np.array(obs_target_prediction[best_act])
            else:
                target = reward
            # Fit predicted value of previous action in previous observation to target value of Bellman equation
            prev_obs_eval_prediction[prev_act] = target

            x_batch.append(prev_obs)
            for act_idx in range(self.n_actions):
                y_batch[act_idx].append(prev_obs_eval_prediction[act_idx])

        if len(x_batch) != 0:
            x_batch = np.array(x_batch)
            y_batch = [np.asarray(y) for y in y_batch]
            self.eval_model.fit(x_batch, y_batch, batch_size=self.batch_size, verbose=0)

    def get_greedy_action(self, obs):
        """
        Retrieves the best next action for the current observation according to the eval Deep Q-Network prediction.

        :param obs: Current observation (state representation)

        :return: action: AgentAction which should be chosen next by the agent according to the eval Deep Q-Network
        """
        predictions = self.predict(np.array([obs]))
        action_values = np.concatenate(predictions).ravel()
        greedy_action_index = np.argmax(action_values)
        return index_to_agent_action(greedy_action_index)
