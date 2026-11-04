"""Experiment 11: tabular Q-Learning on a 4x4 grid world."""
import numpy as np

rng = np.random.default_rng(42)
states, actions = 16, 4  # actions: up, down, left, right
Q = np.zeros((states, actions))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.1, 3000

def next_state(state, action):
    row, col = divmod(state, 4)
    if action == 0: row = max(0, row - 1)
    elif action == 1: row = min(3, row + 1)
    elif action == 2: col = max(0, col - 1)
    else: col = min(3, col + 1)
    return row * 4 + col

for _ in range(episodes):
    state = int(rng.integers(0, 15))
    for _ in range(100):
        action = int(rng.integers(0, actions)) if rng.random() < epsilon else int(np.argmax(Q[state]))
        new_state = next_state(state, action)
        reward = 100 if new_state == 15 else -1
        Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
        state = new_state
        if state == 15:
            break

policy_symbols = np.array(["U", "D", "L", "R"])
print("Q Table:\n", np.round(Q, 2))
print("Optimal policy (goal is G):")
for state in range(states):
    print("G" if state == 15 else policy_symbols[np.argmax(Q[state])], end=" ")
    if state % 4 == 3:
        print()

