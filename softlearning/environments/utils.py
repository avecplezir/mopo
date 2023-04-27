from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

import mopo.env as env_overwrite
import pdb

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}

# class GymAdapter_POMDP(GymAdapter):
#     def __init__(self, domain, task, **environment_params):
#         super().__init__(domain, task, **environment_params)
#         # self.active_observation_shape = 12
#         # setattr(self, 'active_observation_shape', 12)
#
#     def _get_obs(self):
#         obs = self.super()._get_obs()
#         print('HalfCheetahEnv', obs.shape)
#         return obs


# ADAPTERS = {
#     'gym': GymAdapter_POMDP,
# }


def get_environment(universe, domain, task, environment_params):
    if domain in env_overwrite:
        print('[ environments/utils ] WARNING: Using overwritten {} environment'.format(domain))
        env = env_overwrite[domain]()
        env = ADAPTERS[universe](None, None, env=env)
    else:
        env = ADAPTERS[universe](domain, task, **environment_params)
    return env


def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()
    print('environment_kwargs', environment_kwargs)

    return get_environment(universe, domain, task, environment_kwargs)
