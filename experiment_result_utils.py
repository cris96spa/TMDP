import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import math
import mlflow
from mlflow.tracking import MlflowClient
import optuna
import os
import torch
from policy_utils import *

def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def save(path, dict):
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(path, "wb") as f:  # Use "wb" mode for writing binary files
        torch.save(dict, f)
    print(f"Dictionary saved at {path}")

def load(path):
    dict = torch.load(path)
    print(f"Dictionary loaded from {path}")
    return dict
    
def save_to_mlflow(dict, name="Dict"):
    temp_path = "./results.pth"
    save(temp_path, dict)
    mlflow.log_artifact(temp_path)
    os.remove(temp_path)
    print("Dictionary saved to MLflow and local file removed")


def load_from_mlflow(artifact_uri):
    # Download the artifact from MLflow
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    # Load the artifact into memory
    dict = load(local_path)
    print("Dictionary loaded from MLflow")
    return dict

def get_nested_runs(experiment_id):
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    parent_runs = [run for run in runs if run.data.tags.get('mlflow.parentRunId') is not None]
    return parent_runs

def get_nested_artifacts(experiment_id):
    nested_runs = get_nested_runs(experiment_id)
    client = MlflowClient()
    artifacts = []
    for run in nested_runs:
        base_uri = run.info.artifact_uri
        resource_uri = client.list_artifacts(run.info.run_id)[0].path
        artifact_uri = base_uri+"/"+resource_uri
        result = load_from_mlflow(artifact_uri)
        artifacts.append(result)

    return artifacts

def get_parent_runs(experiment_id):
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    parent_runs = [run for run in runs if run.data.tags.get('mlflow.parentRunId') is None]
    return parent_runs

def get_parent_artifacts(experiment_id):
    parent_runs = get_parent_runs(experiment_id)
    client = MlflowClient()
    artifacts = []
    print(parent_runs)
    for run in parent_runs:
        base_uri = run.info.artifact_uri
        print(client.list_artifacts(run.info.run_id))
        list_artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id) if artifact.path.endswith(".pth")]
        if len(list_artifacts) > 0:
            resource_uri = list_artifacts[0]
            artifact_uri = base_uri+"/"+resource_uri
            result = load_from_mlflow(artifact_uri)
            artifacts.append(result)
    return artifacts


"""
    Test policies on the original problem
    Args:
        - tmdp (TMDP): the teleporting MDP
        - thetas (ndarray): collection of parameter vectors associated to the policies
        - episodes (int): number of episodes to run
        - temp (float): temperature value
    return (ndarray): the average reward collected over trajectories for each policy
"""
def test_policies(tmdp:TMDP, thetas, episodes=100, temp=1e-5):    
    rewards = []
    tau = tmdp.tau
    
    for theta in thetas:
        pi = get_softmax_policy(theta, temperature=temp)
        tmdp.reset()
        tmdp.update_tau(0.)
        episode = 0
        cum_r = 0
        traj_count = 0
        while episode < episodes:
            s = tmdp.env.s
            a = select_action(pi[s])
            s_prime, reward, flags, prob = tmdp.step(a)
            cum_r += reward
            if flags["done"]:
                break
                tmdp.reset()
                traj_count += 1
            episode += 1
        cum_r = cum_r/traj_count if traj_count > 0 else cum_r
        rewards.append(cum_r)
    
    tmdp.update_tau(tau)
    return rewards


def test_Q_policies(tmdp:TMDP, Qs, episodes=100):    
    rewards = []
    tau = tmdp.tau
    
    for Q in Qs:
        pi = get_policy(Q)
        tmdp.reset()
        tmdp.update_tau(0.)
        episode = 0
        cum_r = 0
        traj_count = 0
        while episode < episodes:
            s = tmdp.env.s
            a = select_action(pi[s])
            s_prime, reward, flags, prob = tmdp.step(a)
            cum_r += reward
            if flags["done"]:
                break
                tmdp.reset()
                traj_count += 1
            episode += 1
        cum_r = cum_r/traj_count if traj_count > 0 else cum_r
        rewards.append(cum_r)
    
    tmdp.update_tau(tau)
    return rewards


def get_artifacts_from_experiment(experiment_id):
    runs = mlflow.search_runs(experiment_ids=experiment_id)
    artifacts = []
    for run_id in runs["run_id"]:
        artifacts.append(mlflow.get_artifact_uri(run_id=run_id))
    return artifacts