# Policy Gradient

Policy gradient is one of the well-known group in reinforcement learning family. The advantage of policy gradient over the typical Q-learning approach is perform continuous action directly without compute time of backpropagation. Unlike value-based approach such as Q-learning which is output descrete action value, the policy gradient output the continuous actions directly instead of state-action value. Other than that, Q-learning algorithm aimed to minimize the difference between the target Q-values and the Q values predicted by the network, whereas policy gradient aimed to determines the actions by using a softmax output and maximize the expected sum of rewards.

## Summary of policy gradient's characteristics:
* Directly output probability of action selection to learn the optimal policy without the Q function
* Able to handle the continuous large action space


## References:
1. http://karpathy.github.io/2016/05/31/rl/