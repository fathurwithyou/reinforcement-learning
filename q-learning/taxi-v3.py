import gym
import numpy as np
import random

np.bool8 = np.bool_

env = gym.make("Taxi-v3", render_mode="ansi")

alpha = 0.9
gamma = 0.95
epsilon = 1
epsilon_decay = 0.9995
min_epsilon = 0.1
# [(25*5*4), 6] = 500 states, 6 actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))
epochs = 1000
max_steps = 100

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