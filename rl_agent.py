import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.8, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state, :])  # Exploit

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)
        self.exploration_rate = max(0.01, self.exploration_rate * (1 - self.exploration_decay))

    def get_q_table(self):
      return self.q_table