from fanorona_aec import fanorona_v2

env = fanorona_v2.env()
env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()

    if terminated or truncated:
        action = None
    else:
        # random policy
        mask = obs["action_mask"]
        action = env.action_space(agent).sample(mask=mask)

    env.step(action)
    env.render()
env.close()