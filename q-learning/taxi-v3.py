import gym
import numpy as np
import random

np.bool8 = np.bool_

env = gym.make("Taxi-v3", render_mode="ansi")

alpha = 0.9 # Learning rate -> 90% of the error is used to update the Q-value
gamma = 0.95 # Discount factor -> 95% of the future reward is considered
epsilon = 1 # Initial exploration rate -> 100% exploration
epsilon_decay = 0.9995 # Decay rate for exploration -> 0.05% decay per epoch
min_epsilon = 0.1 # Minimum exploration rate -> 10% exploration
# [(25*5*4), 6] = 500 states, 6 actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))
epochs = 1000 # Number of epochs to train the agent
max_steps = 100 # Maximum steps per epoch

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample() # Explore: choose a random action
    else:
        return np.argmax(q_table[state, :]) # Exploit: choose the best action based on Q-table
    
for epoch in range(epochs):
    state, _ = env.reset()
    done = False
    
    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error
        state = next_state
        if done:
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon
    
    if (epoch+1) % 1000 == 0:
        print("Q-table at epoch", epoch, ":\n", q_table)
        
print("Training completed.")

for i in range(10):
    state, _ = env.reset()
    done = False
    print(f"Episode {i+1}:")
    
    while not done:
        action = np.argmax(q_table[state])  
        next_state, reward, done, _, _ = env.step(action)
        print(env.render())
        state = next_state
        if done:
            print(f"Finished with reward: {reward}")
            break