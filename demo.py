from fanorona_aec import fanorona_v3

env = fanorona_v3.env()
seed = None
env.reset(seed=seed)

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    assert obs is not None

    if terminated or truncated:
        action = None
    else:
        # random policy
        mask = obs["action_mask"]
        action = env.action_space(agent).sample(mask=mask)
    env.step(action)
    env.render()
env.close()
