# Actor-Critic

The basic Actor-Critic architecture has two major components; the actor network (also known as poli-cy network) and the critic network. 

Why Actor-Critic as we have DQN?
Actually the "Actor" has previous identity named as policy gradient, which can perform continuous action selection, while Q-learning incapable to do so. The "Critic" part is value-based learning component such as Q-learning.

The Actor perform action based on probablities instead of epsilon-greedy strategy.