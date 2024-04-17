import os
import sys
import argparse

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import numpy.matlib as matl
import scipy.io as io
import matplotlib.pyplot as plt
import copy

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

from dynamics.freeflyer_obs import compute_constraint_to_go, compute_reward_to_go
from optimization.ff_scenario_obs import robot_radius, safety_margin, dataset_scenario

parser = argparse.ArgumentParser(description='transformer-ff')
parser.add_argument('--data_dir', type=str, default='dataset/' + dataset_scenario,
                    help='defines directory from where to load files')
parser.add_argument('--data_dir_torch', type=str, default='dataset/' + dataset_scenario + '/torch/v05',  # remember to change v0# depending on the dataset version loaded
                    help='defines directory from where to load files')
args = parser.parse_args()
args.data_dir = root_folder + '/' + args.data_dir
args.data_dir_torch = root_folder + '/' + args.data_dir_torch

print('Loading data...', end='')

data_scp = np.load(args.data_dir + '/dataset-ff-v05-scp.npz')
data_cvx = np.load(args.data_dir + '/dataset-ff-v05-cvx.npz')
data_param = np.load(args.data_dir + '/dataset-ff-v05-param.npz') 

# Pre-compute torch states roe and rtn
states_scp = data_scp['states_scp']
print('States shapes', states_scp.shape[0])
torch_states_scp = torch.from_numpy(states_scp)
torch.save(torch_states_scp, args.data_dir_torch + '/torch_states_scp.pth')

states_cvx = data_cvx['states_cvx']
torch_states_cvx = torch.from_numpy(states_cvx)
torch.save(torch_states_cvx, args.data_dir_torch + '/torch_states_cvx.pth')

# Pre-compute torch observations
observations_scp = data_scp['observations_scp']
torch_observations_scp = torch.from_numpy(observations_scp)
torch.save(torch_observations_scp, args.data_dir_torch + '/torch_observations_scp.pth')

observations_cvx = data_cvx['observations_cvx']
torch_observations_cvx = torch.from_numpy(observations_cvx)
torch.save(torch_observations_cvx, args.data_dir_torch + '/torch_observations_cvx.pth')

# Pre-compute torch actions
actions_scp = data_scp['actions_scp']
torch_actions_scp = torch.from_numpy(actions_scp)
torch.save(torch_actions_scp, args.data_dir_torch + '/torch_actions_scp.pth')

actions_cvx = data_cvx['actions_cvx']
torch_actions_cvx = torch.from_numpy(actions_cvx)
torch.save(torch_actions_cvx, args.data_dir_torch + '/torch_actions_cvx.pth')

# Pre-compute torch rewards to go and constraints to go
torch_rtgs_scp = torch.from_numpy(compute_reward_to_go(actions_scp))
torch.save(torch_rtgs_scp, args.data_dir_torch + '/torch_rtgs_scp.pth')

torch_rtgs_cvx = torch.from_numpy(compute_reward_to_go(actions_cvx))
torch.save(torch_rtgs_cvx, args.data_dir_torch + '/torch_rtgs_cvx.pth')

n_obs = data_param['n_obs']
obs_position = data_param['obs_position']
obs_radius = data_param['obs_radius']

# obs = copy.deepcopy(obs)
# obs['radius'] = (obs['radius'] + robot_radius) * safety_margin
obs_radius = (obs_radius + robot_radius) * safety_margin
torch_ctgs_scp = torch.from_numpy(compute_constraint_to_go(states_scp, obs_position, obs_radius, n_obs))
torch.save(torch_ctgs_scp, args.data_dir_torch + '/torch_ctgs_scp.pth')

torch_ctgs_cvx = torch.from_numpy(compute_constraint_to_go(states_cvx, obs_position, obs_radius, n_obs))
torch.save(torch_ctgs_cvx, args.data_dir_torch + '/torch_ctgs_cvx.pth')

# Permutation
if states_cvx.shape[0] != states_scp.shape[0]:
    raise RuntimeError('Different dimensions of cvx and scp datasets.')
perm = np.random.permutation(states_cvx.shape[0] * 2)
np.save(args.data_dir_torch + '/permutation.npy', perm)

print('Completed\n')

# IDEA to speed up things if needed

# def do_preprocessing(states_rtn, actions, oe, dt, n_data, n_time):

#     constraint_to_go = np.empty(shape=(n_data, n_time), dtype=float)
#     rewards_to_go = np.empty(shape=(n_data, n_time), dtype=float)
#     stm_roe  = np.empty((n_data, n_time, 6, 6))
#     cim_roe  = np.empty((n_data, n_time, 6, 3))
#     stm_rtn  = np.empty((n_data, n_time, 6, 6))
#     cim_rtn  = np.empty((n_data, n_time, 6, 3))

#     for n in range(n_data):

#         constr_koz_n, constr_koz_violation_n = check_koz_constraint(np.transpose(np.squeeze(states_rtn[n, :, :])), n_time)
#         r_tot_n = np.sum(la.norm(actions[n, :, :], axis=1))

#         for t in range(n_time):

#             constraint_to_go[n, t] = np.sum(constr_koz_violation_n[t:])
#             rewards_to_go[n, t] = -np.sum(la.norm(actions[n, t:, :], axis=1)) / r_tot_n
#             stm_roe[n,t] = state_transition(oe[n, t], dt[n])
#             cim_roe[n,t] = control_input_matrix(oe[n, t])
#             map_t = map_mtx_roe_to_rtn(oe[n, t])
#             if t < n_time - 1:
#                 map_t_new = map_mtx_roe_to_rtn(oe[n, t+1])
#             else: 
#                 a = oe[n, t][0]
#                 nn = np.sqrt(mu_E/a**3)
#                 oe_new = oe[n, t] + np.array([0, 0, 0, 0, 0, nn*dt.item(n)]).reshape((6,))
#                 map_t_new = map_mtx_roe_to_rtn(oe_new)
#             stm_rtn[n,t] = np.matmul(map_t_new, np.matmul(stm_roe[n,t], np.linalg.inv(map_t)))
#             cim_rtn[n,t] = np.matmul(map_t, cim_roe[n,t])

#     return constraint_to_go
