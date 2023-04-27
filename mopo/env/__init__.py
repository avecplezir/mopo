import sys

from .ant import AntEnv
from .humanoid import HumanoidEnv
from .halfcheetah import HalfCheetahEnv

# env_overwrite = {'Ant': AntEnv, 'Humanoid': HumanoidEnv}
# env_overwrite = {'Ant': AntEnv}
# env_overwrite = {}
env_overwrite = {'HalfCheetah': HalfCheetahEnv}

sys.modules[__name__] = env_overwrite