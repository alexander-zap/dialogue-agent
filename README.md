# dialogue-agent

## Content
* [What is the goal of dialogue-agent?](#what-is-the-goal-of-dialogue-agent)
* [Why Reinforcement Learning?](#why-reinforcement-learning)
* [Component Overview](#component-overview)
* [Installation](#installation)
* [Technologies](#technologies)


## What is the goal of dialogue-agent?
dialogue-agent is a framework which uses reinforcement learning to train a goal-oriented agent for dialogue use cases involving real users.
The final agent will help users achieve a specific goal e.g. booking a movie ticket which corresponds to the communicated criteria.
For this the agent communicates with the user in a turn-based manner to ask and respond to their utterances as well as ask further questions to determine the wishes of the user.

Features of the dialogue-agent:
- Selects next action to be executed by the agent which has the highest probability of being a successful step towards achieving the user-goal (based on previous experiences)
- Includes prior conversation history in his decision process
- Learns behavior without
  - prior knowledge of the "correct" or best behavior
  - manual specification of rules
  - providing training data
- Can continue learning while in productive use to dynamically react to changes in user trend


## Why Reinforcement Learning?
With the goal of learning a "mapping" from dialogue state to the best agent response, one might think of using standard supervised machine learning approaches.
These require pre-annotated data (dialogue state paired with the "correct"/best response), which in turn requires the knowledge of the best response policy in the first place.

Problems with this approach:
- Defining the best (most successful) behavior beforehand is difficult
- The behavior might need to change over time depending on the user trend

These problems are solved with reinforcement learning.
Reinforcement learning trains an agent through trial-and-error conversations with simulated users.
In this process the agent learns the best policy (of choosing the appropriate response to answer to the user). 
Once the agent shows promising performance with simulated users, it can be deployed to real users and has the option to continue learning in order to further improve its performance or adjust its policy to changed user behavior.

The main differences of reinforcement learning to supervised learning can be seen in the data generation.
While supervised machine learning needs pre-annotated data (annotated with the "correct"/best behavior), reinforcement learning is able to learn the best policy from trial-and-error.
Training data is generated from a simulated or real interaction with a user, with "goodness" of the behavior being assessed by a reward function (e.g. positive reward for successful recommendation, negative reward for wrong recommendation).


## Component Overview
### User
- Has a user-goal
- Every turn informs one aspect of his user-goal and requests unknown information from the agent

### Agent
- Does not know the user-goal
- Every turn tries to deduce and fulfill user-goal by executing one action from multiple possible actions, including
  - Informing user about requested information
  - Requesting specific information from the user
  - Predicting and suggesting user-goal once enough information has been collected

### Database
- Database containing all possible entries of the product the user is trying to get (e.g. available movies)
- Agent retrieves requested information from here

### Dialogue State
- Represents the state of the dialogue (including history of agent/user actions) and database


## Installation
1. Clone this repository
2. Install all required Python packages from the requirements.txt
3. Execute the main.py-file to start the showcase. The showcase consists of training an agent in the use case of movie-booking.


## Technologies
The technologies (linked with their respective paper) which have been used for this framework are:
- TC-Bot (https://arxiv.org/abs/1703.01008 and https://arxiv.org/abs/1612.05688)
- Double Deep Q-Networks (https://arxiv.org/abs/1509.06461)
- Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)
- Deep Ordinal Reinforcement Learning (https://arxiv.org/abs/1905.02005)

