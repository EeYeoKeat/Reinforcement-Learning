from medium_qlearning_env import Env
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# import environment
env = Env()

# QTable : initialize table with the Q-Values for every (state,action) pair
qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

# hyperparameters
epochs = 100     #default epochs value is 50
gamma = 0.1     #default gamma value is 0.1
epsilon = 0.08  #default epsilon value is 0.08
decay = 0.1     #default decay value is 0.1
alpha = 0.1		#learning rate

total_steps = []

# training loop
for i in range(1, epochs):
    state, reward, done = env.reset()
    steps = 0

    while not done:
        os.system('cls')
        print("epoch #", i, "/", epochs)
        env.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.randomAction()
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))

        # take action
        next_state, reward, done = env.step(action)

        # update qtable value with Bellman equation
        qtable[state][action] += alpha * (reward + (gamma * max(qtable[next_state])-qtable[state][action]))

       # update state 
        state = next_state
    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon

    total_steps.append(steps)

    print(f"Completed with {steps} steps")
    print(total_steps)
    time.sleep(0.8)

print(qtable)

plt.plot(range(1, epochs), total_steps)
plt.xlabel('epochs')
plt.ylabel('total steps')
plt.title('Total steps taken in each epoch')
plt.show()


