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
        self.target_model = self.build_model(target=True)

    # Creates big neural net for DQN
    def build_model(self, target=False):
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
        # copy evaluation model to target model at first replay and then every 200 replay steps
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
            # fit predicted value of previous action in previous observation to target value of Bellman equation
            prev_obs_eval_prediction[prev_act] = target

            x_batch.append(prev_obs)
            for act_idx in range(self.n_actions):
                y_batch[act_idx].append(prev_obs_eval_prediction[act_idx])

        if len(x_batch) != 0:
            x_batch = np.array(x_batch)
            y_batch = [np.asarray(y) for y in y_batch]
            self.eval_model.fit(x_batch, y_batch, batch_size=self.batch_size, verbose=0)

    def get_greedy_action(self, obs):
        predictions = self.predict(np.array([obs]))
        action_index = np.argmax(predictions)
        return index_to_agent_action(action_index)

    @staticmethod
    def reward_to_ordinal(reward):
        if reward < -1:
            return 0
        elif reward == -1:
            return 1
        elif reward > -1:
            return 2
