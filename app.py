import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time
from rl_agent import QLearningAgent #replace your_rl_module with the file name.

# --- App Title ---
st.title("Reinforcement Learning Demo: Frozen Lake")

# --- Parameters ---
st.sidebar.header("Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.8)
discount_factor = st.sidebar.slider("Discount Factor", 0.8, 1.0, 0.95)
exploration_decay = st.sidebar.slider("Exploration Decay", 0.001, 0.01, 0.005, step=0.001)
episodes = st.sidebar.slider("Episodes", 100, 5000, 1000)

# --- Frozen Lake Environment (Example) ---
def create_frozen_lake(size=4):
    lake = np.random.choice(['F', 'H', 'S', 'G'], size=(size, size), p=[0.7, 0.1, 0.1, 0.1])
    lake[0, 0] = 'S'  # Start
    lake[size - 1, size - 1] = 'G'  # Goal
    return lake

def get_state(lake, position):
    return position[0] * lake.shape[1] + position[1]

def get_position(state, size):
    return state // size, state % size

def step(lake, position, action):
    row, col = position
    size = lake.shape[0]

    if action == 0:  # Left
        col = max(0, col - 1)
    elif action == 1:  # Down
        row = min(size - 1, row + 1)
    elif action == 2:  # Right
        col = min(size - 1, col + 1)
    elif action == 3:  # Up
        row = max(0, row - 1)

    new_position = (row, col)
    new_state = get_state(lake, new_position)

    if lake[row, col] == 'H':
        reward = -1
        done = True
    elif lake[row, col] == 'G':
        reward = 1
        done = True
    else:
        reward = 0
        done = False

    return new_state, new_position, reward, done

# --- Training ---
lake = create_frozen_lake()
state_size = lake.shape[0] * lake.shape[1]
action_size = 4  # Left, Down, Right, Up

agent = QLearningAgent(state_size, action_size, learning_rate, discount_factor, exploration_decay=exploration_decay)

st.subheader("Training Progress")
progress_bar = st.progress(0)
q_table = None

for episode in range(episodes):
    position = (0, 0)
    state = get_state(lake, position)
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, next_position, reward, done = step(lake, position, action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        position = next_position
    progress_bar.progress((episode + 1) / episodes)
    q_table = agent.get_q_table()

st.success("Training Complete!")

# --- Visualization ---
st.subheader("Agent's Policy")

def visualize_policy_arrows(lake, q_table):
    size = lake.shape[0]
    policy = np.zeros_like(lake, dtype=int)
    for i in range(size):
        for j in range(size):
            state = get_state(lake, (i, j))
            policy[i, j] = np.argmax(q_table[state, :])

    fig, ax = plt.subplots()
    # Corrected line: create a float array of zeros
    ax.imshow(np.zeros_like(lake, dtype=float), cmap='gray', alpha=0.1)
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(size):
        for j in range(size):
            action = policy[i, j]
            dx, dy = 0, 0
            if action == 0:  # Left
                dx = -0.3
            elif action == 1:  # Down
                dy = 0.3
            elif action == 2:  # Right
                dx = 0.3
            elif action == 3:  # Up
                dy = -0.3
            ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    st.pyplot(fig)

visualize_policy_arrows(lake, q_table)