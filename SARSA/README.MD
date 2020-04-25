# State–action–reward–state–action (SARSA)


In SARSA, the agent will take action before learning update iteration. The difference from Q-learning is the Q-learning agent will only estimate the maximum value of action from epsilon-greedy policy in next state, but it not neccessay will take that specific maximum value action. While SARSA agent will confirm to take the action under epsilon-greedy policy. 