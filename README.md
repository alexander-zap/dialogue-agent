# dialogue-agent
## What is the goal of dialogue-agent?
dialogue-agent is a framework which uses reinforcement learning to train a goal-oriented agent for dialogue use cases involving real users.
The final agent will help users achieve a specific goal e.g. booking a movie ticket which corresponds to the communicated criteria.
For this the agent communicates with the user in a turn-based manner to ask and respond to their utterances as well as ask further questions to determine the wishes of the user.


## Why Reinforcement Learning?
With the goal of learning a "mapping" from dialogue state to the best agent response, one might think of using standard supervised machine learning approaches.
These require pre-annotated data (dialogue state paired with the "golden" best response), which in turn requires the knowledge of the best response policy in the first place.

Problems with this approach:
- Defining the perfect behavior beforehand is difficult
- Perfect (most successful) behavior might need to change over time depending on the user

These problems are solved with Reinforcement Learning.
Reinforcement Learning trains an agent through trial-and-error conversations with simulated users.
In this process the agent learns the best policy (of choosing the appropriate response to answer to the user). 
Once the agent shows promising performance with simulated users, it is deployed to real users and has the option to continue learning in order to further improve its performance or adjust its policy to changed user behavior.


## Component Overview
### User
- has a user-goal
- informs about his wishes and requests information from the agent

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

