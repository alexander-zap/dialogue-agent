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
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def update(self, prev_obs, prev_act, obs, reward, done, warm_up=False, replay=True):
        self.remember(prev_obs, prev_act.feasible_action_index, obs, reward, done)
        if replay and not warm_up and len(self.memory) > self.batch_size:
            self.replay()

    def remember(self, prev_obs, prev_act_index, obs, rew, d):
        self.memory.append((prev_obs, prev_act_index, obs, rew, d))

    def replay(self):
        # copy evaluation model to target model at first replay and then every 200 replay steps
        if self.replay_counter % self.replace_target_iter == 0:
            self.target_model.set_weights(self.eval_model.get_weights())
        self.replay_counter += 1

        # FIXME: Training on all training data doesn't seem right to me (leave this currently so it is close to GO-Bot)
        num_batches = len(self.memory) // self.batch_size
        for b in range(num_batches):
            mini_batch = random.sample(self.memory, self.batch_size)
            x_batch, y_batch = [], []
            prev_obs_batch = np.array([sample[0][0] for sample in mini_batch])
            obs_batch = np.array([sample[2][0] for sample in mini_batch])
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
                x_batch.append(prev_obs[0])
                y_batch.append(prediction)
            self.eval_model.fit(np.array(x_batch), np.array(y_batch), batch_size=self.batch_size, verbose=0)

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

    def get_greedy_action(self, obs):
        action_index = np.argmax(self.eval_model.predict(obs)[0])
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
