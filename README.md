# dialogue-agent
## What is the goal of dialogue-agent?
dialogue-agent is a framework which uses reinforcement learning to train a goal-oriented agent for dialogue use cases involving real users.
The final agent will help users achieve a specific goal e.g. booking a movie ticket which corresponds to the communicated criteria.
For this the agent communicates with the user in a turn-based manner to ask and respond to their utterances as well as ask further questions to determine the wishes of the user.


## Why Reinforcement Learning?


## Component Overview
### User
- has a user goal
- informs about his wished and requests information from the agent

### Database
- database containing all possible entries of the product the user is trying to get (e.g. available movies)

### Dialogue State
- represents the state of the dialogue (including history of agent/user actions) and database

### Agent
- tries to fulfill user goal by requesting missing information
- informs user about requested information
- executes or suggests predicted user-goal

## Installation
1. Clone this repository
2. Install all required Python packages from the requirements.txt
3. Execute the main.py-file for start the showcase. The showcase consists of training an agent in the use case of movie-booking.

## Technologies
The technologies (linked with their respective paper) which have been used for this framework are:
- TC-Bot (https://arxiv.org/abs/1703.01008 and https://arxiv.org/abs/1612.05688)
- Double Deep Q-Networks (https://arxiv.org/abs/1509.06461)
- Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)
- Deep Ordinal Reinforcement Learning (https://arxiv.org/abs/1905.02005)

