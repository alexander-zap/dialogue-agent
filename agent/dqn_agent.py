from agent.agent import Agent
from util_functions import index_to_agent_action
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
        model = Sequential()
        model.add(Dense(20, input_dim=self.input_size, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

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
        x_batch, y_batch = [], []
        prev_obs_batch = np.array([sample[0] for sample in mini_batch])
        obs_batch = np.array([sample[2] for sample in mini_batch])
        prev_obs_eval_prediction_batch = self.eval_model.predict(prev_obs_batch)
        obs_eval_prediction_batch = self.eval_model.predict(obs_batch)
        obs_target_prediction_batch = self.target_model.predict(obs_batch)
        for i, (prev_obs, prev_act_index, obs, rew, d) in enumerate(mini_batch):
            prev_obs_eval_prediction = prev_obs_eval_prediction_batch[i]
            if not d:
                best_act = np.argmax(obs_eval_prediction_batch[i])
                target = rew + self.gamma * obs_target_prediction_batch[i][best_act]
            else:
                target = rew
            # Fit predicted value of previous action in previous observation to target value of Bellman equation
            prev_obs_eval_prediction[prev_act_index] = target
            x_batch.append(prev_obs)
            y_batch.append(prev_obs_eval_prediction)
        self.eval_model.fit(np.array(x_batch), np.array(y_batch), batch_size=self.batch_size, verbose=0)

    def get_greedy_action(self, obs):
        """
        Retrieves the best next action for the current observation according to the eval Deep Q-Network prediction.

        :param obs: Current observation (state representation)

        :return: action: AgentAction which should be chosen next by the agent according to the eval Deep Q-Network
        """
        obs = obs.reshape(1, -1)
        action_index = np.argmax(self.eval_model.predict(obs)[0])
        return index_to_agent_action(action_index)
