from fanorona_aec import fanorona_v1

env = fanorona_v1.env()
env.reset()
env.step(352)
env.render()
print(env.state().legal_moves)
print(env.observations['player_0']['action_mask'].nonzero())
env.step(518)