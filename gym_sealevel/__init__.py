from gym.envs.registration import register

register(
    id='sealevel-v0',
    entry_point='gym_sealevel.envs:SealevelEnv',
)