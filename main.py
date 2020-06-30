import json
import pickle
from numpy import mean
from agent.dqn_agent import DQNAgent
from user.usersim_rulebased import RulebasedUsersim
from dialog_config import feasible_agent_actions
from state_tracker import StateTracker


class Chatbot:
    def __init__(self):

        # Load dict of possible slot values for every slot type (from movie_dict.txt)
        movie_dict = json.load(open("resources/movie_dict.json", "r", encoding="utf-8"))
        # Load database of movies
        database = pickle.load(open("resources/movie_db.pkl", "rb"), encoding="latin1")

        # Create state tracker
        self.state_tracker = StateTracker(database)

        # Create user with list of user goals
        self.user = RulebasedUsersim(json.load(open("resources/movie_user_goals.json", "r", encoding="utf-8")))

        # Create agent
        self.agent = DQNAgent(alpha=0.001, gamma=0.9, epsilon=0.0, epsilon_min=0.0,
                              n_actions=len(feasible_agent_actions), observation_dim=(StateTracker.state_size()),
                              batch_size=16, memory_len=500000, replace_target_iter=200, replay_iter=100)

    def run(self, n_episodes, step_size=100, warm_up=False):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
        TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

        """

        batch_episode_rewards = []
        batch_successes = []
        batch_success_best = 0.0

        for episode in range(n_episodes):

            self.episode_reset()
            # print("########################\n------ EPISODE {} ------\n########################".format(episode))
            done = False
            success = 0
            episode_reward = 0

            replay = episode % self.agent.replay_iter == 0
            prev_observation = self.state_tracker.get_state()
            # 1) Agent takes action given state tracker's representation of dialogue (state)
            prev_agent_action = self.agent.choose_action(prev_observation, warm_up=warm_up)
            while not done:
                # 2) 3) 4) 5) 6)
                # print("--- Round {} ---".format(prev_agent_action.round_num))
                # print(prev_agent_action)
                observation, reward, done, success = self.env_step(prev_agent_action)

                self.agent.update(prev_observation, prev_agent_action, observation, reward, done,
                                  warm_up=warm_up, replay=replay)
                self.user.turn += 1
                self.agent.turn += 1
                episode_reward += reward

                agent_action = self.agent.choose_action(observation, warm_up=warm_up)
                prev_observation = observation
                prev_agent_action = agent_action
                replay = False

            if not warm_up:
                self.agent.end_episode(n_episodes)
            # Evaluation
            batch_episode_rewards.append(episode_reward)
            batch_successes.append(success)
            # print("--- Episode {} End: Success {} ---".format(episode, success))

            if episode % step_size == 0:
                # Check success rate
                success_rate = mean(batch_successes)
                avg_reward = mean(batch_episode_rewards)

                print('Episode: {} SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate,
                                                                           avg_reward))
                if success_rate > batch_success_best and not warm_up and episode > 1000:
                    print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate,
                                                                                        avg_reward))
                    batch_success_best = success_rate
                    self.agent.empty_memory()
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


if __name__ == "__main__":
    chatbot = Chatbot()
    print("########################\n--- STARTING WARM UP ---\n########################")
    chatbot.run(n_episodes=1000, warm_up=True)
    print("########################\n--- STARTING TRAINING ---\n#########################")
    chatbot.run(n_episodes=10000, warm_up=False)
