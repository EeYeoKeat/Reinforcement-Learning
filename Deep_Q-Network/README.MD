# Deep Q-Network (DQN)

In naive model, the model suffered from few issues and the learning is not stable. The reason causing instability is the correlations present in the sequence of observations.

In the DQN model, the problems are solved by these significant components:
<ol>
   <li>Experience Replay</li>
   <li>Q-target Network</li>
</ol>

Both evaluate network and target network is initialized with same parameter. The DQN agent only will start learning when it obtain a batch of memory. The transition will be stored in replay memory in order to perform experience replay. The agent will sample random minibatch of transition from memory to perform gradient descent step with respect to the network parameters. The random sampling from batch replay memory to update the target network will break the correlations between the observation sequence and smoothing over changes in the data distribution.

In addition, although the evaluate network will be updated iteratively by calculating the loss value between the target and evaluate network, but the target network parameters will be only periodically updated with parameters from evaluate network that adjusts the action-values (Q) towards target values, thereby also reducing correlations with the target.


## References

<ol>
<li><a href="https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf">Human-level control through deep reinforcement learning (Volodymyr Mnih et al., 2015)</a></li>
</ol>