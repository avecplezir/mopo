import argparse
from distutils.util import strtobool
import json
import os
import pickle

import numpy as np
import tensorflow as tf

import mopo.static
from mopo.models.bnn import BNN
from mopo.models.fake_env import FakeEnv
from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from dogo.results import get_experiment_details
from softlearning.replay_pools import SimpleReplayPool


# TODO: Confirm these parameters are appropriate in this case
PARAMETERS_PATH = "/home/ajc348/rds/hpc-work/mopo/dogo/bnn_params.json"
EPISODE_LENGTH = 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_checkpoint_path',
                        type=str,
                        help='Path to the policy checkpoint.')
    parser.add_argument('dynamics_experiment',
                        type=str,
                        help='Experiment whose dynamics model should be used.')
    # parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', 'None'),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()
    
    if args.render_mode == 'None':
        args.render_mode = None

    return args

# TODO: determine the kwargs being passed into fake_env, and whether to keep them
def rollout_model(policy, fake_env, gym_env, **kwargs):
    pool = SimpleReplayPool(gym_env.observation_space, gym_env.action_space, EPISODE_LENGTH)
    obs = gym_env.convert_to_active_observation(gym_env.reset())[None,:]

    infos = []
    for _ in range(EPISODE_LENGTH):
        act = policy.actions_np(obs)
        next_obs, rew, term, info = fake_env.step(obs, act, **kwargs)
        pol = np.zeros((len(obs), 1))
        infos.append(info)
        samples = {'observations': obs, 'actions': act, 'next_observations': next_obs, 'rewards': rew, 'terminals': term, 'policies': pol}
        pool.add_samples(samples)
        obs = next_obs
    
    path = pool.batch_by_indices(
        np.arange(pool._size),
        observation_keys=getattr(gym_env, 'observation_keys', None))
    path['infos'] = infos

    return path

def rollouts(n_paths, policy, fake_env, evaluation_environment):
    paths = [rollout_model(policy, fake_env, evaluation_environment) for _ in range(n_paths)]
    return paths

def simulate_policy(args):
    session = tf.keras.backend.get_session()

    policy_checkpoint_path = args.policy_checkpoint_path.rstrip('/')
    policy_experiment_path = (
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(policy_checkpoint_path)))))
    )

    variant_path = os.path.join(policy_experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(policy_checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    # Load the dynamics model
    exp_details = get_experiment_details(args.dynamics_experiment, get_elites=True)
    model_dir = os.path.join(exp_details.results_dir, 'models')
    with open(PARAMETERS_PATH, 'r') as f:
        dynamics_params = json.load(f)
    dynamics_params["model_dir"] = model_dir

    dynamics_model = BNN(dynamics_params)
    dynamics_model.model_loaded = True
    dynamics_model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    dynamics_model._model_inds = exp_details.elites

    domain = environment_params['domain']
    static_fns = mopo.static[domain.lower()]
    fake_env = FakeEnv(
        dynamics_model, static_fns, penalty_coeff=0.,
        penalty_learned_var=True, gym_environment=evaluation_environment
    )

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(
            args.num_rollouts,
            policy,
            fake_env,
            evaluation_environment,
        )

    #### print rewards
    rewards = [path['rewards'].sum() for path in paths]
    print('Rewards: {}'.format(rewards))
    print('Mean: {}'.format(np.mean(rewards)))
    ####
    
    return paths


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)
