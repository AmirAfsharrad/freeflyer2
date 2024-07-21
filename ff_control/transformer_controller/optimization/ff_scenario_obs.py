import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np

# Problem dimensions
N_STATE = 6
N_ACTION = 3
N_CLUSTERS = 4
N_OBS_MAX = 4
SINGLE_OBS_DIM = 3

# time problem constants
S = 101  # number of control switches
n_time_rpod = S - 1

# constants
mass = 16.0
inertia = 0.18
robot_radius = 0.15
F_max_per_thruster = 0.2
thrusters_lever_arm = 0.11461
Lambda = np.array([[0, 1, 0, 1],
                   [1, 0, 1, 0],
                   [thrusters_lever_arm, -thrusters_lever_arm, -thrusters_lever_arm, thrusters_lever_arm]])
Lambda_inv = np.array([[0, 0.5, 1 / (4 * thrusters_lever_arm)],
                       [0.5, 0, -1 / (4 * thrusters_lever_arm)],
                       [0, 0.5, -1 / (4 * thrusters_lever_arm)],
                       [0.5, 0, 1 / (4 * thrusters_lever_arm)]])

dataset_scenario = 'var_obstacles_5_scenarios_1'
# Table, start and goal regions dimensions
table = {
    'xy_low': np.array([0., 0.]),
    'xy_up': np.array([3.5, 2.5])
}
start_region = {
    'xy_low': table['xy_low'] + robot_radius,
    'xy_up': np.array([0.5, 2.5]) - robot_radius
}
goal_region = {
    'xy_low': np.array([3.0, 0.]) + robot_radius,
    'xy_up': table['xy_up'] - robot_radius
}
obs_region = {
    'xy_low': np.array([0.5, 0.]),
    'xy_up': np.array([3.0, 2.5])
}

# Obstacle
T = 40.0  # max final time horizon in sec
dt = T / n_time_rpod
#

# n_obs = obs['position'].shape[0]

safety_margin = 1.1

# PID Controller
control_period = 0.1
gain_f = 2.0 * 0
gain_df = 10.0 * 2
gain_t = 0.2 * 0
gain_dt = 0.4 * 2
K = np.array([[gain_f, 0, 0, gain_df, 0, 0],
              [0, gain_f, 0, 0, gain_df, 0],
              [0, 0, gain_t, 0, 0, gain_dt]])

# Optimization interface
iter_max_SCP = 20
trust_region0 = 10.
trust_regionf = 0.005
J_tol = 10 ** (-6)

obs1 = {
    'position': np.array([[1.0, 0.7],
                          [1.5, 1.7],
                          [2.5, 0.75],
                          [2.5, 1.75]]),
    'radius': np.array([0.18, 0.15, 0.12, 0.2])
}

obs2 = {
    'position': np.array([[.8, .5],
                          [1.3, 1.7],
                          [2, 1.05],
                          [2.6, 2.05]]),
    'radius': np.array([0.17, 0.2, 0.14, 0.13])
}

obs3 = {
    'position': np.array([[1.5, .6],
                          [2.1, 1.13],
                          [1.1, 1.45],
                          [1.9, 2.05]]),
    'radius': np.array([0.138, 0.11, 0.15, 0.19])
}


obs4 = {
    'position': np.array([[1, 1],
                          [1.75, 1.5],
                          [2.5, 2],
                          [2.4, .45]]),
    'radius': np.array([0.123, 0.151, 0.131, 0.184])
}

obs5 = {
    'position': np.array([[2.41969267, 0.8912164], 
                          [0.94172849, 0.46758919], 
                          [2.22846824, 2.35803776], 
                          [0.86247335, 2.26967807]]),
    'radius': np.array([0.19225326, 0.11313881, 0.10559907, 0.18528049])
}

obs6 = {
    'position': np.array([[1.62689314, 1.98860954],
                        [0.88884833, 0.75615519],
                        [2.23800377, 0.45431808],
                        [2.06970026, 1.1674797 ]]),
    'radius': np.array([0.10816975, 0.12672767, 0.11910985, 0.13555548])
}

obs7 = {
    'position': np.array([[0.85370543, 0.99110425],
                        [0.84359307, 0.28624518],
                        [2.49302603, 2.19911688],
                        [2.48829184, 0.7253271 ]]),
    'radius': np.array([0.15110599, 0.10785146, 0.19881381, 0.12667007])
}

obs8 = {
    'position': np.array([[0.67002279, 0.75223312],
                        [2.64512784, 0.88091674],
                        [1.17944753, 1.98158626],
                        [0.78448421, 1.36134232]]),
    'radius': np.array([0.12461378, 0.13683983, 0.11705715, 0.12039186])
}

obs9 = {
    'position': np.array([[1.97157936, 1.23268527],
                        [1.24549497, 1.85344324],
                        [2.44949628, 0.19396605],
                        [2.63707386, 2.25951662]]),
    'radius': np.array([0.15943664, 0.11171497, 0.10946028, 0.1661533 ])
}

obs10 = {
    'position': np.array([ [2.1873858648612146, 1.4195382901329396],
                        [1.513435454186436, 0.5164013896529343],
                        [1.1072122389517203, 2.2404009719543145],
                        [1.14933076823709, 1.1482519970784735]]),
    'radius': np.array([ 0.18372475290849227, 0.11456488453657399, 0.1529624400595343, 0.12791000716427708])
}

obs11 = {
    'position': np.array([ [1.322254215490553, 1.8963692433323505],
                        [2.40300029441051, 0.37469491290811524],
                        [1.8152717303515336, 0.8562467429718106],
                        [2.6020031227216416, 1.2843504140538304]]),
    'radius': np.array([ 0.15614304519088035, 0.15518705647808573, 0.10193008422083, 0.11564739755372316])
}

obs12 = {
    'position': np.array([ [1.8515708055695828, 0.6612535915615683],
                        [1.273302738904587, 1.0542174587757076],
                        [2.5707656352427, 1.285562098877053],
                        [2.192820108276734, 2.027946857026107]]),
    'radius': np.array([ 0.1439455760282616, 0.15681929106417777, 0.1860621880796965, 0.17652270525313737])
}

obs13 = {
    'position': np.array([ [1.6727547666191276, 1.2843406322905162],
                        [1.0840661300637007, 0.2361579292884463],
                        [1.3801072724175363, 1.9784890280459697],
                        [2.727819438528107, 1.1777855304127303]]),
    'radius': np.array([ 0.15339303661300935, 0.16598369582592476, 0.10075645289194371, 0.15256289935419431])
}

obs14 = {
    'position': np.array([ [0.710380480642965, 0.4092006848297754],
                        [2.4469923000972282, 1.3751889153503096],
                        [0.9936296633677815, 1.5303459521132994],
                        [2.7757534693567227, 0.5409475594002084]]),
    'radius': np.array([ 0.1887091905061234, 0.18756636434405327, 0.13668096096033766, 0.18544470821811299])
}

obs_list = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9, obs10, obs11, obs12, obs13, obs14]
n_obs_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]