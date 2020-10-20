from agent.agent import Agent
from util_functions import index_to_agent_action, raw_agent_action_to_index
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dialog_config import agent_rule_requests


class DQNAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval_model = self.build_model()
        self.target_model = self.build_model()

    # Creates neural net for DQN
    def build_model(self):
        model = Sequential()
        model.add(Dense(80, input_dim=self.input_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def replay(self):
        # copy evaluation model to target model at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replay_counter += 1

        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch, y_batch = [], []
        prev_obs_batch = np.array([sample[0] for sample in mini_batch])
        obs_batch = np.array([sample[2] for sample in mini_batch])
        prev_obs_prediction_batch = self.eval_model.predict(prev_obs_batch)
        obs_prediction_batch = self.target_model.predict(obs_batch)
        for i, (prev_obs, prev_act, obs, rew, d) in enumerate(mini_batch):
            prediction = prev_obs_prediction_batch[i]
            if not d:
                best_act = np.argmax(prev_obs_prediction_batch[i])
                target = rew + self.gamma * obs_prediction_batch[i][best_act]
            else:
                target = rew
            # fit predicted value of previous action in previous observation to target value of max_action
            prediction[prev_act] = target
            x_batch.append(prev_obs)
            y_batch.append(prediction)
        self.eval_model.fit(np.array(x_batch), np.array(y_batch), batch_size=self.batch_size, verbose=0)

    def get_greedy_action(self, obs):
        obs = obs.reshape(1, -1)
        action_index = np.argmax(self.eval_model.predict(obs)[0])
        return index_to_agent_action(action_index)
