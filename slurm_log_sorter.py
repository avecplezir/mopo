import os
import re
import sys
import json
from glob import glob


############################################################################
# SLURM logs are saved to the `slurm_logs` directory and the base directory.
# Experiment results are stored in the `ray_mopo` directory.
# This script moves the SLURM logs to the correct experiment directory in
# `ray_mopo`.
############################################################################


def main(results_dir: str):    
    ######################################################################    
    # Experiment State Files
    # Generated by Ray/MOPO, these live one directory above the experiment
    # directory they pertain to. Move them down to the correct experiment
    # directory.
    ######################################################################
    # Identify all of the experiment state files
    es_paths = glob(os.path.join(results_dir, 'experiment_state-*.json'))

    # Loop through these
    for es_path in es_paths:
        # Load the file
        with open(es_path, 'r') as f:
            try:
                exp_state = json.load(f)
            except json.decoder.JSONDecodeError as e:
                # These sometimes have trailing closing curly brackets, or other quirks.
                # Remove these to obtain valid JSON files that can be loaded.
                print(es_path)
                raise e

        # Identify the correct experiment directory
        es_logdir = os.path.abspath(exp_state['checkpoints'][-1]['logdir'])

        # Move the file to the correct experiment directory
        os.rename(es_path, os.path.join(es_logdir, os.path.basename(es_path)))
    
    ############
    # SLURM Logs
    ############
    # Identify all of the training log files
    train_log_paths = glob(os.path.join('slurm_logs', 'train-log.*'))

    # Loop through the files
    for train_log_path in train_log_paths:
        train_log_path = os.path.abspath(train_log_path)

        # Identify the machine and SLURM files
        # SLURM saves these to the base directory of the repo
        job_id = os.path.basename(train_log_path).split('.')[-1]
        train_log_name = f'train-log.{job_id}'
        machine_file_name = f'machine.file.{job_id}'
        slurm_file_name = f'slurm-{job_id}.out'

        # Identify the experiment directory the files should be moved to
        log_dir_res = []
        with open(train_log_path, 'r') as f:
            for l in f:
                log_dir_res = re.findall("(?<=\[ MOPO \] log_dir: ).*(?= \|)", l)
                if len(log_dir_res) > 0:
                    break
        
        # In addition to training environment models and policies we also use the HPC to score models
        # and run policies in environments. We will not be able to identify an experiment directory for
        # these (as there isn't one), and so they are moved to `slurm_logs/no_log_dir`. They are then
        # periodically deleted (manually).
        if len(log_dir_res) == 0:
            print(f'Could not find log_dir in {train_log_path}')
            os.rename(os.path.join('slurm_logs', train_log_name), os.path.join('slurm_logs', 'no_log_dir', train_log_name))
            os.rename(machine_file_name, os.path.join('slurm_logs', 'no_log_dir', machine_file_name))
            os.rename(slurm_file_name, os.path.join('slurm_logs', 'no_log_dir', slurm_file_name))
            continue
        log_dir = log_dir_res[0]

        if os.path.dirname(log_dir) != os.path.abspath(results_dir):
            # Only move the log files which match the directory of interest
            # Ignore the others - re-run this script specifically for those directories
            continue

        # Move the log files to the correct directory
        os.rename(os.path.join('slurm_logs', train_log_name), os.path.join(log_dir, train_log_name))
        os.rename(machine_file_name, os.path.join(log_dir, machine_file_name))
        os.rename(slurm_file_name, os.path.join(log_dir, slurm_file_name))

    ######################################################
    # Check that each folder has exactly one expected file
    # Never experienced this check to have failed
    ######################################################
    exp_results_dirs = glob(os.path.join(results_dir, 'seed:*'))
    for exp_results_dir in exp_results_dirs:
        assert len(glob(os.path.join(exp_results_dir, 'train-log.*'))) == 1
        assert len(glob(os.path.join(exp_results_dir, 'machine.file.*'))) == 1
        assert len(glob(os.path.join(exp_results_dir, 'slurm-*.out'))) == 1

if __name__ == "__main__":
    # results_dir = sys.argv[1]
    results_dir = os.path.abspath('ray_mopo/HalfCheetah/halfcheetah_mixed_rt_4_101e3')
    main(results_dir)
