# Multi-Armed Bandit Problem

The multi-armed bandit (MAB) problem often refer to arm of a K-slot machine to pull for maximize the total reward in a series of trials. It can model the real-world learning and optimization problems. In marketing terms, a multi-armed bandit solution is a smarter or more complex version of A/B testing that uses machine learning algorithms to dynamically allocate traffic to variations that are performing well, while allocating less traffic to variations that are underperforming.

In reinforcement learning (RL), the agent always need to balance between exploitation and exploration in order to learn the optimal target values. Therefore, some mechanism are proposed to deal with this exploration-exploitation dilemma.

Algorithms:
1) Epsilon-greedy policy
2) Softmax exploration
3) Upper confidence bound (UCB)
4) Thompson sampling

1) The epsilon-greedy policy is commonly applied in several RL approaches such as Q-learning and DQN. With the epsilon-greedy policy, the agent are allowed to explore the environment with a probability of epsilon to select random actions, otherwise, it will select the maximum value with a probability 1-epsilon.

2) Softmax exploration, also known as Boltzmann exploration, is another strategy used for finding an optimal bandit. In the epsilon-greedy policy, all of the actions are given chance equivalently, but in softmax exploration, the action are selected based on a probability from the Boltzmann distribution.

<img src="https://github.com/ee2110/Reinforcement-Learning/blob/master/img/Probability%20boltzmann%20distribution.JPG">

3) Instead of choosing actions based on probability, the upper confidence bound (UCB) selecting the best action based on a confidence interval. The UCB will more focus on explore the actions that has a high UCB rather than other.

4) Thompson sampling (TS) is another popularly used algorithm to overcome the exploration-exploitation dilemma. It is a probabilistic algorithm and is based on a prior distribution. The term prior distribution indicate that it is initial distributions calculated based on prior mean sample rewards for each of the k arms. Using Bernoulli rewards, the value of beta distribution [alpha, beta] lies within the interval [0,1]. Alpha represents the number of times we receive the positive rewards and beta represents the number of times we receive the negative rewards. Sample a value from each of the k distributions and use this value as a prior
mean, then the arm that has the highest prior mean and observes the reward. Lastly, use the observed reward to modify the prior distribution.


References:
1) https://www.optimizely.com/optimization-glossary/multi-armed-bandit/
2) Vermorel J., Mohri M. (2005) Multi-armed Bandit Algorithms and Empirical Evaluation. In: Gama J., Camacho R., Brazdil P.B., Jorge A.M., Torgo L. (eds) Machine Learning: ECML 2005. ECML 2005. Lecture Notes in Computer Science, vol 3720. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11564096_42
3) Hands-On Reinforcement Learning with Python: Master reinforcement and deep reinforcement learning using OpenAI Gym and TensorFlow, by Sudharsan RavichandiranJun 2018, Packt Publishing Ltd