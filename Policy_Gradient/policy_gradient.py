import numpy as np

class PolicyGradient_Agent():
    def __init__(self, alpha, gamma, n_actions):
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.n_actions = n_actions
        
        
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0
                
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.n_actions)])
        else:
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)]) #if can, better reindex the action before choose
            action = np.argmax(actions)
        return action        
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.min_epsilon else self.min_epsilon
        
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)
        
        self.Q[(state, action)] += self.learning_rate * (reward + self.discount_factor*self.Q[(state_,a_max)]-self.Q[(state, action)])
        
        self.decrement_epsilon()
