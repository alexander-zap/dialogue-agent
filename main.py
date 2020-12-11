import json
import pickle

from numpy import mean

from agent.dqn_agent_split_action_nets import DQNAgent
from dialog_config import feasible_agent_actions
from state_tracker import StateTracker
from user.usersim_rulebased import RulebasedUsersim


class Chatbot:

    def __init__(self):
        # Load database of movies (if you get an error unpickling movie_db.pkl then run pickle_converter.py)
        database = pickle.load(open("resources/movie_db.pkl", "rb"), encoding="latin1")

        # Create state tracker
        self.state_tracker = StateTracker(database)

        # Create user with list of user goals
        self.user = RulebasedUsersim(json.load(open("resources/movie_user_goals.json", "r", encoding="utf-8")))

        # Create agent
        self.agent = DQNAgent(alpha=0.001, gamma=0.9, epsilon=0.5, epsilon_min=0.05,
                              n_actions=len(feasible_agent_actions), n_ordinals=3,
                              observation_dim=(StateTracker.state_size()),
                              batch_size=256, memory_len=80000, prioritized_memory=False,
                              replay_iter=16, replace_target_iter=200)

    def run(self, n_episodes, step_size=100, success_rate_threshold=0.4, warm_up=False):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented chatbot task (except warm_up, which fills memory with rule-based behavior)
        Training of the agent's neural network occurs every episode that step_size is a multiple of.
        Replay memory is flushed every time a best success rate is recorded, starting with success_rate_threshold.
        Terminates when the episode reaches n_episodes.

        """

        batch_episode_rewards = []
        batch_successes = []
        batch_success_best = 0.0
        step_counter = 0

        for episode in range(n_episodes):

            # print("########################\n------ EPISODE {} ------\n########################".format(episode))
            self.episode_reset()
            done = False
            success = False
            episode_reward = 0

            # Initialize episode with first user and agent action
            prev_observation = self.state_tracker.get_state()
            prev_agent_action = self.agent.choose_action(prev_observation, warm_up=warm_up)
            while not done:
                step_counter += 1
                replay = step_counter % self.agent.replay_iter == 0
                # print(prev_agent_action)
                # 2) 3) 4) 5) 6)
                observation, reward, done, success = self.env_step(prev_agent_action)
                self.agent.update(prev_observation, prev_agent_action, observation, reward, done,
                                  warm_up=warm_up, replay=replay)
                # 1) Agent takes action given state tracker's representation of dialogue (observation)
                agent_action = self.agent.choose_action(observation, warm_up=warm_up)

                episode_reward += reward
                prev_observation = observation
                prev_agent_action = agent_action

            if not warm_up:
                self.agent.end_episode(n_episodes)

            # Evaluation
            # print("--- Episode: {} SUCCESS: {} REWARD: {} ---".format(episode, success, episode_reward))
            batch_episode_rewards.append(episode_reward)
            batch_successes.append(success)
            if episode % step_size == 0:
                # Check success rate
                success_rate = mean(batch_successes)
                avg_reward = mean(batch_episode_rewards)

                print('Episode: {} SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate,
                                                                           avg_reward))
                if success_rate > batch_success_best and not warm_up and success_rate > success_rate_threshold:
                    print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate,
                                                                                        avg_reward))
                    self.agent.empty_memory()
                    batch_success_best = success_rate
                batch_successes = []
                batch_episode_rewards = []

    def env_step(self, agent_action):
        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.get_action(agent_action)
        # print(user_action)
        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)
        # 6) Get next state and add experience
        observation = self.state_tracker.get_state(done)
        return observation, reward, done, True if success is 1 else False

    def episode_reset(self):
        # Reset the state tracker
        self.state_tracker.reset()
        # Reset the user
        self.user.reset()
        # Reset the agent
        self.agent.turn = 0
        # User start action
        user_action, _, _, _ = self.user.get_action(None)
        # print(user_action)
        self.state_tracker.update_state_user(user_action)


if __name__ == "__main__":
    chatbot = Chatbot()
    print("########################\n--- STARTING WARM UP ---\n########################")
    chatbot.run(n_episodes=0, warm_up=True)
    print("########################\n--- STARTING TRAINING ---\n#########################")
    chatbot.run(n_episodes=10000, warm_up=False, success_rate_threshold=0.25)
