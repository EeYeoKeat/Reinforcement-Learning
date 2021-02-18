# DRL approaches performance in Enduro-v0

This folder consists of many type of DRL approache algorithms and evaluated in OpenAI Gym environment, more specific, Atari 2600 games.

## What is Enduro-v0?
Enduro-v0 is one of the Atari environment and the main objective of this game is to maximize the score in game. In the environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3). Each action is repeatedly performed for a duration of k frames, where k is uniformly sampled from {2,3,4}.

Since the screen images as input, we will also using Convolutional Neural Network (CNN) model to digest the input screen as input state.

Comparison of performance among the DRL approaches. <br>

| DRL Algorithms 		| Min. Score   | Max. Score  | Mean. Score |
|     :---:      		|    :---:     |    :---:    |    :---:    |
| Deep Q-Network (DQN)  	| nan          | nan         | nan         |
| git diff       		| git diff     | nan         | nan         |