# Naive Deep Q-Network 

The DQN model used in this work is linear neural network without experience replay for batch update.
The agent did not perform as good as expected real DQN model.

From the graph we can see the performance is not consistent, when agent achieved high score in previous steps, it then drop to almost zero score just like it totally new start learn from zero.

Problems that the agent encountered in this continuous state space:
<ol>
  <li>The agent have learned to achieve higher score but it discard the experience after that</li>
  <li>The exploration of the agent at the environment is insufficient due to epsilon decrease too fast</li>
  <li>Agent is chasing moving target as it use the same network to evaluate the maximum action and to select the maximum action. The network is updated every step</li>
</ol>