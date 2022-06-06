import os
import json
from typing import List

import numpy as np
import tensorflow as tf

from mopo.models.bnn import BNN
from mopo.models.constructor import format_samples_for_training
from mopo.models.fake_env import FakeEnv
from mopo.off_policy.loader import restore_pool_contiguous
from softlearning.environments.utils import get_environment_from_params
from softlearning.replay_pools.utils import get_replay_pool_from_variant


# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-02_18-00-39fnhnkljn/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-02_18-00-52d5tl1m3e/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-02_18-03-01g6l6v5s7/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-02_18-03-07tlo_lnqh/models"

# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1234_2022-06-02_19-31-12zvhbtily/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1234_2022-06-02_19-31-11xl42u_po/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1234_2022-06-02_19-34-59z5a4yar7/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1234_2022-06-02_19-35-33js3upueh/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:4321_2022-06-02_19-37-14a2q9o806/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:4321_2022-06-02_19-37-59u77cj7_f/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:4321_2022-06-02_19-39-52sn2zawbk/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:4321_2022-06-02_19-40-08sckp5u_v/models"

# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-03_16-05-26sitz4u3t/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-03_16-06-26h_fbckpg/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_pap5_101e3/seed:1443_2022-06-03_16-08-02zd2g2vw7/models"



# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_10-12-32frkill7m/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_10-13-01h3k6c1g5/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_10-14-117mj_xfq5/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_10-14-47qm688i6a/models"

# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1234_2022-06-03_10-15-49g6s6nvdn/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1234_2022-06-03_10-16-211wgd0h6a/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1234_2022-06-03_10-17-06x84f8rk2/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1234_2022-06-03_10-18-14jzadorh6/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:4321_2022-06-03_10-26-23i2iqe13u/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:4321_2022-06-03_10-26-55_34mzot9/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:4321_2022-06-03_10-28-01akst9xyh/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:4321_2022-06-03_10-28-58fur7k1s_/models"

# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_16-10-43c99ekxmd/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_16-17-26ryn7rqqd/models"
# MODEL_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/HalfCheetah/halfcheetah_d3rlpy_mp1_101e3/seed:1443_2022-06-03_16-18-51x87sdp5y/models"

DATA_PATHS = [
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1_100000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P0-3.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P1-4.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P0_25000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P1_25000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P2_25000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P3_25000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P4_25000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P0_100000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P1_100000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P2_100000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P3_100000.npy",
    "/home/ajc348/rds/hpc-work/dogo_results/data/D3RLPY-MP1-P4_100000.npy",
]

PARAMETERS_PATH = "/home/ajc348/rds/hpc-work/mopo/dogo/bnn_params.json"
OUTPUT_BASE_DIR = "/home/ajc348/rds/hpc-work/dogo_results/mopo/model_scoring"


def score_model(model_dir: str, data_paths: List[str], deterministic=True):
    # Load parameters - the mininal needed give we are loading a pre-trained model
    with open(PARAMETERS_PATH, 'r') as f:
        params = json.load(f)
    params["model_dir"] = model_dir

    # Set the seed
    seed = params['seed']
    if seed is not None:
        np.random.seed(seed)

    ##########################
    # Get training environment
    ##########################
    # From: examples/development/main.py

    training_environment = get_environment_from_params(params['environment_params']['training'])

    ###################
    # Instantiate model
    ###################
    # This will load the model whose location is specified in the parameters
    # From: /rds/user/ajc348/hpc-work/mopo/mopo/models/constructor.py

    bnn = BNN(params)
    bnn.model_loaded = True
    bnn.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

    for data_path in data_paths:
        #####################################
        # Create and load replay pool/dataset
        #####################################
        # From: examples/development/main.py

        params["replay_pool_params"]["pool_load_path"] = data_path
        replay_pool = get_replay_pool_from_variant(params, training_environment)

        # Usually called by restore_pool, which is called in mopo/algorithms/mopo.py
        restore_pool_contiguous(replay_pool, params['replay_pool_params']['pool_load_path'])
        env_samples = replay_pool.return_all_samples()
        
        ###################################
        # Run obs and actions through model
        ###################################
        # From the `step` method in FakeEnv: mopo/models/fake_env.py

        obs, acts, rewards, next_obs = env_samples['observations'], env_samples['actions'], env_samples['rewards'], env_samples['next_observations']
        inputs = np.concatenate((obs, acts), axis=-1)
        outputs = np.concatenate((rewards, next_obs), axis=-1)

        ensemble_model_means, ensemble_model_vars = bnn.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)
    
        ensemble_reward_model_means, ensemble_next_obs_model_means = ensemble_model_means[:,:,:1], ensemble_model_means[:,:,1:]
        ensemble_reward_model_vars, ensemble_next_obs_model_vars = ensemble_model_vars[:,:,:1], ensemble_model_vars[:,:,1:]

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        samples = np.mean(ensemble_samples, axis=0)
        pred_rewards, pred_next_obs = samples[:,:1], samples[:,1:]

        k = outputs.shape[-1]
        ## [ num_networks, batch_size ]
        log_probs = -1/2 * (k * np.log(2*np.pi) + np.log(ensemble_model_vars).sum(-1) + (np.power(outputs-ensemble_model_means, 2)/ensemble_model_vars).sum(-1))
        prob = np.exp(log_probs).mean()
        log_prob = np.log(prob)

        reward_log_probs = -1/2 * (1 * np.log(2*np.pi) + np.log(ensemble_reward_model_vars).sum(-1) + (np.power(rewards-ensemble_reward_model_means, 2)/ensemble_reward_model_vars).sum(-1))
        reward_prob = np.exp(reward_log_probs).mean()
        reward_log_prob = np.log(reward_prob)

        next_obs_log_probs = -1/2 * ((k-1) * np.log(2*np.pi) + np.log(ensemble_next_obs_model_vars).sum(-1) + (np.power(next_obs-ensemble_next_obs_model_means, 2)/ensemble_next_obs_model_vars).sum(-1))
        next_obs_prob = np.exp(next_obs_log_probs).mean()
        next_obs_log_prob = np.log(next_obs_prob)
        
        ###############
        # Determine MSE
        ###############
        actual_rewards, actual_next_obs = env_samples['rewards'], env_samples['next_observations']
        reward_mse = ((actual_rewards-pred_rewards)**2).mean()
        obs_mse = ((actual_next_obs-pred_next_obs)**2).mean()

        print(f'Model Directory: {params["model_dir"]}')
        print(f'Data Used: {params["replay_pool_params"]["pool_load_path"]}')
        print(f'Reward MSE: {reward_mse} | Observation MSE: {obs_mse} | log_prob: {log_prob}')

        json_results = {
            "model_dir": model_dir,
            "data_path": data_path,
            "seed": seed,
            "deterministic": deterministic,
            "reward_mse": float(reward_mse),
            "observation_mse": float(obs_mse),
            "log_prob": float(log_prob),
            "reward_log_prob": float(reward_log_prob),
            "next_obs_log_prob": float(next_obs_log_prob),
        }
        json_output_dir = os.path.join(OUTPUT_BASE_DIR, model_dir.split("/")[-3], model_dir.split("/")[-2])
        if not os.path.isdir(json_output_dir):
            os.makedirs(json_output_dir)
        json_output_path = os.path.join(json_output_dir, f'{data_path.split("/")[-1][:-4]}_{seed}.json')
        with open(json_output_path, 'w') as f:
            json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    score_model(MODEL_DIR, DATA_PATHS)
