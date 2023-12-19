import numpy as np
from fanorona_aec import fanorona_v3


class QLearningAgent:
    def __init__(
        self,
        action_space,
        q_table=None,  # to allow "weight sharing"
        seed=None,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
    ):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table
        self.prng = np.random.default_rng(seed)

    def get_action(self, state, mask):
        if self.prng.random() < self.exploration_rate:
            return self.action_space.sample(mask=mask)  # Explore (random action)
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


def main():
    env = fanorona_v3.env()

    agents = {
        agent: QLearningAgent(env.action_space(agent)) for agent in env.possible_agents
    }

    num_episodes = 1000
    decay_rate = 0.99

    for episode in range(num_episodes):
        env.reset(seed=42)

        for agent in env.agent_iter():
            env.render()
            obs, reward, termination, truncation, info = env.last()
            obs, mask = obs["observation"], obs["action_mask"]

            if termination or truncation:
                action = None
            else:
                action = agents[agent].get_action(obs, mask)

            env.step(action)

            if not termination and not truncation:
                (
                    next_obs,
                    next_reward,
                    next_termination,
                    next_truncation,
                    next_info,
                ) = env.last()
                next_obs, next_mask = next_obs["observation"], next_obs["action_mask"]
                agents[agent].update(obs, action, reward, next_obs)

        for agent in agents.values():
            agent.decay_exploration_rate(decay_rate)
    env.close()


if __name__ == "__main__":
    main()
