from gymnasium.envs.registration import register

register(
    id='HumanoidG-v0',
    entry_point='custom_envs.envs:HumanoidEnv',
)

register(
    id='HumanoidLeg-v0',
    entry_point='custom_envs.envs:HumanoidLegEnv',
)

register(
    id='HumanoidG-v2',
    entry_point='custom_envs.envs:HumanoidEnv_v2',
)

register(
    id='HumanoidG-v3',
    entry_point='custom_envs.envs:HumanoidEnv_v3',
)

register(
    id='HumanoidG-v4',
    entry_point='custom_envs.envs:HumanoidEnv_v4',
)

register(
    id='Centipede-v0',
    entry_point='custom_envs.envs:CentipedeEnv',
)

register(
    id='Centipede-v2',
    entry_point='custom_envs.envs:CentipedeEnv_v2',
)

register(
    id='Dog-v0',
    entry_point='custom_envs.envs:DogEnv',
)

register(
    id='Dog-v2',
    entry_point='custom_envs.envs:DogEnv_v2',
)

register(
    id='Dog-v3',
    entry_point='custom_envs.envs:DogEnv_v3',
)

register(
    id='HalfCheetahCustom-v0',
    entry_point='custom_envs.envs:HalfCheetahEnv',
)

register(
    id='Dog2-v0',
    entry_point='custom_envs.envs:Dog2Env',
)

register(
    id='Dog2-v2',
    entry_point='custom_envs.envs:Dog2Env_v2',
)

register(
    id='Centipede8-v2',
    entry_point='custom_envs.envs:Centipede8Env_v2',
)

register(
    id='Dog8_2-v0',
    entry_point='custom_envs.envs:Dog8_2Env',
)

register(
    id='Dog8_2-v2',
    entry_point='custom_envs.envs:Dog8_2Env_v2',
)

register(
    id='Dog18_2-v0',
    entry_point='custom_envs.envs:Dog18_2Env',
)

register(
    id='Dog18_2-v2',
    entry_point='custom_envs.envs:Dog18_2Env_v2',
)

register(
    id='Centipede4_2-v0',
    entry_point='custom_envs.envs:Centipede4_2Env',
)

register(
    id='Centipede4_2-v2',
    entry_point='custom_envs.envs:Centipede4_2Env_v2',
)