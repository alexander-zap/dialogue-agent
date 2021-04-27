import json
import pickle

from numpy import mean

from agent import DQNAgentSplit
from dialog_config import feasible_agent_actions
from gui.chat_application import ChatApplication
from state_tracker import StateTracker
from user import User, RulebasedUsersim


class Dialogue:

    def __init__(self, load_agent_model_from_directory: str = None):
        # Load database of movies (if you get an error unpickling movie_db.pkl then run pickle_converter.py)
        database = pickle.load(open("resources/movie_db.pkl", "rb"), encoding="latin1")

        # Create state tracker
        self.state_tracker = StateTracker(database)

        # Create user simulator with list of user goals
        self.user_simulated = RulebasedUsersim(
            json.load(open("resources/movie_user_goals.json", "r", encoding="utf-8")))

        # Create GUI for direct text interactions
        self.gui = ChatApplication()

        # Create user instance for direct text interactions
        self.user_interactive = User(nlu_path="user/regex_nlu.json", use_voice=False, gui=self.gui)

        # Create empty user (will be assigned on runtime)
        self.user = None

        # Create agent
        self.agent = DQNAgentSplit(alpha=0.001, gamma=0.9, epsilon=0.5, epsilon_min=0.05,
                                   n_actions=len(feasible_agent_actions), n_ordinals=3,
                                   observation_dim=(StateTracker.state_size()),
                                   batch_size=256, memory_len=80000, prioritized_memory=True,
                                   replay_iter=16, replace_target_iter=200)
        if load_agent_model_from_directory:
            self.agent.load_agent_model(load_agent_model_from_directory)

    def run(self, n_episodes, step_size=100, warm_up=False, interactive=False, learning=True):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented dialog task (except warm_up, which fills memory with rule-based behavior)
        Training of the agent's neural network occurs every episode that step_size is a multiple of.
        Replay memory is flushed every time a best success rate is recorded, starting with success_rate_threshold.
        Terminates when the episode reaches n_episodes.

        """

        if interactive:
            self.user = self.user_interactive
            self.gui.window.update()
        else:
            self.user = self.user_simulated

        if not learning:
            self.agent.epsilon = 0.0

        batch_episode_rewards = []
        batch_successes = []
        batch_success_best = 0.0
        step_counter = 0

        for episode in range(n_episodes):

            # print("########################\n------ EPISODE {} ------\n########################".format(episode))
            self.episode_reset(interactive)
            done = False
            success = False
            episode_reward = 0

            # Initialize episode with first user and agent action
            prev_observation = self.state_tracker.get_state()
            # 1) Agent takes action given state tracker's representation of dialogue (observation)
            prev_agent_action = self.agent.choose_action(prev_observation, warm_up=warm_up)
            while not done:
                step_counter += 1
                # 2) 3) 4) 5) 6a)
                observation, reward, done, success = self.env_step(prev_agent_action, interactive)
                if learning:
                    replay = step_counter % self.agent.replay_iter == 0
                    # 6b) Add experience
                    self.agent.update(prev_observation, prev_agent_action, observation, reward, done,
                                      warm_up=warm_up, replay=replay)
                # 1) Agent takes action given state tracker's representation of dialogue (observation)
                agent_action = self.agent.choose_action(observation, warm_up=warm_up)

                episode_reward += reward
                prev_observation = observation
                prev_agent_action = agent_action

            if not warm_up and learning:
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
                if success_rate > batch_success_best and learning and not warm_up:
                    print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate,
                                                                                        avg_reward))
                    self.agent.save_agent_model()
                    batch_success_best = success_rate
                batch_successes = []
                batch_episode_rewards = []

        if learning and not warm_up:
            # Save final model
            self.agent.save_agent_model()

    def env_step(self, agent_action, interactive=False):
        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        if interactive:
            self.gui.insert_message(agent_action.to_utterance(), "Shop Assistant")
        # print(agent_action)
        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.get_action(agent_action)
        # print(user_action)
        # 4) Infuse error into user action (currently inactive)
        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)
        # 6a) Get next state
        observation = self.state_tracker.get_state(done)
        return observation, reward, done, True if success is 1 else False

    def episode_reset(self, interactive=False):
        # Reset the state tracker
        self.state_tracker.reset()
        # Reset the user
        self.user.reset()
        # Reset the agent
        self.agent.turn = 0
        # Reset the interactive GUI
        if interactive:
            self.gui.reset_text_widget()
            self.gui.insert_message("Guten Tag! Wie kann ich Ihnen heute helfen?", "Shop Assistant")
        # User start action
        user_action, _, _, _ = self.user.get_action(None)
        # print(user_action)
        self.state_tracker.update_state_user(user_action)


if __name__ == "__main__":
    dialogue = Dialogue()
    print("########################\n--- STARTING WARM UP ---\n########################")
    dialogue.run(n_episodes=4000, warm_up=True)
    print("########################\n--- STARTING TRAINING ---\n#########################")
    dialogue.run(n_episodes=10000, warm_up=False)
