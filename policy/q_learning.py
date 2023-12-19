import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_states,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((num_states, num_actions))

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.num_actions)  # Explore (random action)
        else:
            return np.argmax(self.q_table[state])  # Exploit (best action)

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        target = (
            reward + self.discount_factor * self.q_table[next_state, best_next_action]
        )
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * error  # Update Q-table

    def decay_exploration_rate(self, decay_rate):
        self.exploration_rate *= decay_rate
