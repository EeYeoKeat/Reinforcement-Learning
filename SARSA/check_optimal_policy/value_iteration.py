import gym
import numpy as np

env = gym.make('FrozenLake8x8-v0')

def value_iteration(env, gamma = 1.0):
    # the discount factor value is 1
    # indicate all of the reward in future states are considered
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000
    threshold = 1e-20
    
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        
        for state in range(env.observation_space.n):
            Q_value = []
            
            for action in range(env.action_space.n):
                next_states_rewards = []
                
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
                    
                Q_value.append(np.sum(next_states_rewards))
                
            value_table[state] = max(Q_value)
            
        if (np.sum(np.fabs(updated_value_table - value_table)) <=threshold):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
        
    return value_table

def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy

def test_policy(learned_policy, n_games):
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        scores = []

        for step in range(len(learned_policy)):
            action = int(learned_policy[step])
            observation_, reward, done, info = env.step(action)
            score += reward
            
        scores.append(score)
        
    return scores

optimal_value_function = value_iteration(env=env,gamma=1.0)
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(optimal_policy)

