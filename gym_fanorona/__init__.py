from gym.envs.registration import register

register(
    id="fanorona-v0", entry_point="gym_fanorona.envs:FanoronaEnv",
)
