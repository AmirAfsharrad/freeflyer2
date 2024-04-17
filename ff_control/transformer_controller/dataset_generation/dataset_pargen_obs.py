import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from dynamics.freeflyer_obs import FreeflyerModel, sample_init_target, ocp_no_obstacle_avoidance, \
    ocp_obstacle_avoidance, generate_random_obstacles, generate_perfect_observations
from optimization.ff_scenario_obs import (N_STATE, N_ACTION, N_CLUSTERS, N_OBS_MAX, n_time_rpod, dt, T, S,
                                          dataset_scenario)
import numpy as np
from multiprocessing import Pool, set_start_method
import itertools
from tqdm import tqdm


def for_computation(input):
    # Input unpacking
    current_data_index = input[0]
    other_args = input[1]
    ff_model = other_args['ff_model']

    # Randomic sample of initial and final conditions
    init_state, target_state = sample_init_target()
    obs = generate_random_obstacles(num_obstacles=np.random.randint(1, N_OBS_MAX + 1))

    # Output dictionary initialization
    out = {'feasible': True,
           'states_cvx': [],
           'actions_cvx': [],
           'actions_t_cvx': [],
           'states_scp': [],
           'actions_scp': [],
           'actions_t_scp': [],
           'target_state': [],
           'dtime': [],
           'time': [],
           'obstacles': []
           }

    # Solve simplified problem -> without obstacle avoidance
    traj_cvx_i, J_cvx_i, iter_cvx_i, feas_cvx_i = ocp_no_obstacle_avoidance(ff_model, init_state, target_state, obs)

    if np.char.equal(feas_cvx_i, 'optimal'):

        #  Solve scp with obstacles
        try:
            traj_scp_i, J_scp_i, iter_scp_i, feas_scp_i, = ocp_obstacle_avoidance(ff_model, traj_cvx_i['states'],
                                                                                  traj_cvx_i['actions_G'], init_state,
                                                                                  target_state, obs)

            if np.char.equal(feas_scp_i, 'optimal'):
                # Save cvx and scp problems in the output dictionary
                out['states_cvx'] = np.transpose(traj_cvx_i['states'][:, :-1])
                out['actions_cvx'] = np.transpose(traj_cvx_i['actions_G'])
                out['actions_t_cvx'] = np.transpose(traj_cvx_i['actions_t'])

                out['states_scp'] = np.transpose(traj_scp_i['states'][:, :-1])
                out['actions_scp'] = np.transpose(traj_scp_i['actions_G'])
                out['actions_t_scp'] = np.transpose(traj_scp_i['actions_t'])

                out['target_state'] = target_state
                out['dtime'] = dt
                out['time'] = np.linspace(0, T, S)[:-1]

                out['obstacles'] = obs
            else:
                out['feasible'] = False
        except:
            out['feasible'] = False
    else:
        out['feasible'] = False

    return out


if __name__ == '__main__':

    N_data = 200000
    N_data = 10
    set_start_method('spawn')

    n_S = N_STATE  # state size
    n_A = N_ACTION  # action size
    n_C = N_CLUSTERS  # cluster size
    n_obs_max = N_OBS_MAX  # maximum number of obstacles

    # Model initialization
    ff_model = FreeflyerModel()
    other_args = {
        'ff_model': ff_model
    }

    states_cvx = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float)  # [m,m,m,m/s,m/s,m/s]
    actions_cvx = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float)  # [m/s]
    actions_t_cvx = np.empty(shape=(N_data, n_time_rpod, n_C), dtype=float)

    states_scp = np.empty(shape=(N_data, n_time_rpod, n_S), dtype=float)  # [m,m,m,m/s,m/s,m/s]
    actions_scp = np.empty(shape=(N_data, n_time_rpod, n_A), dtype=float)  # [m/s]
    actions_t_scp = np.empty(shape=(N_data, n_time_rpod, n_C), dtype=float)

    target_state = np.empty(shape=(N_data, n_S), dtype=float)
    dtime = np.empty(shape=(N_data,), dtype=float)
    time = np.empty(shape=(N_data, n_time_rpod), dtype=float)

    n_obs = - np.ones(shape=(N_data,), dtype=int)
    obs_position = np.empty(shape=(N_data, n_obs_max, 2), dtype=float)
    obs_radius = np.empty(shape=(N_data, n_obs_max), dtype=float)

    # For other types of observation (such as a 2d map), the following line should be replaced
    observations = np.empty(shape=(N_data, n_time_rpod, 3 * n_obs_max), dtype=float)

    i_unfeas = []

    # Pool creation --> Should automatically select the maximum number of processes
    p = Pool(processes=24)
    for i, res in enumerate(
            tqdm(p.imap(for_computation, zip(np.arange(N_data), itertools.repeat(other_args))), total=N_data)):
        # for i in np.arange(N_data):
        #    res = for_computation((i, other_args))
        # If the solution is feasible save the optimization output
        if res['feasible']:
            states_cvx[i, :, :] = res['states_cvx']
            actions_cvx[i, :, :] = res['actions_cvx']
            actions_t_cvx[i, :, :] = res['actions_t_cvx']

            states_scp[i, :, :] = res['states_scp']
            actions_scp[i, :, :] = res['actions_scp']
            actions_t_scp[i, :, :] = res['actions_t_scp']

            target_state[i, :] = res['target_state']
            dtime[i] = res['dtime']
            time[i, :] = res['time']

            n_obs[i] = len(res['obstacles']['radius'])
            obs_position[i, :n_obs[i]] = res['obstacles']['position']
            obs_radius[i, :n_obs[i]] = res['obstacles']['radius']

            # For other types of observation (such as a 2d map), the following line should be replaced
            observations[i, :, :3 * n_obs[i]] = generate_perfect_observations(res['obstacles']['position'],
                                                                           res['obstacles']['radius'])

        # Else add the index to the list
        else:
            i_unfeas += [i]

        # if i % 50000 == 0:
        if i % 5 == 0:
            np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-scp' + str(i),
                                states_scp=states_scp, actions_scp=actions_scp, actions_t_scp=actions_t_scp,
                                i_unfeas=i_unfeas)
            np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-cvx' + str(i),
                                states_cvx=states_cvx, actions_cvx=actions_cvx, actions_t_cvx=actions_t_cvx,
                                i_unfeas=i_unfeas)
            np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-param' + str(i),
                                target_state=target_state, time=time, dtime=dtime, n_obs=n_obs,
                                obs_position=obs_position, obs_radius=obs_radius)

    # Remove unfeasible data points
    if i_unfeas:
        states_cvx = np.delete(states_cvx, i_unfeas, axis=0)
        actions_cvx = np.delete(actions_cvx, i_unfeas, axis=0)
        actions_t_cvx = np.delete(actions_t_cvx, i_unfeas, axis=0)

        states_scp = np.delete(states_scp, i_unfeas, axis=0)
        actions_scp = np.delete(actions_scp, i_unfeas, axis=0)
        actions_t_scp = np.delete(actions_t_scp, i_unfeas, axis=0)

        target_state = np.delete(target_state, i_unfeas, axis=0)
        dtime = np.delete(dtime, i_unfeas, axis=0)
        time = np.delete(time, i_unfeas, axis=0)

        n_obs = np.delete(n_obs, i_unfeas, axis=0)
        obs_position = np.delete(obs_position, i_unfeas, axis=0)
        obs_radius = np.delete(obs_radius, i_unfeas, axis=0)

    #  Save dataset (local folder for the workstation)
    np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-scp', states_scp=states_scp,
                        observations_scp=observations, actions_scp=actions_scp, actions_t_scp=actions_t_scp)
    np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-cvx', states_cvx=states_cvx,
                        observations_cvx=observations, actions_cvx=actions_cvx, actions_t_cvx=actions_t_cvx)
    np.savez_compressed(root_folder + '/dataset/' + dataset_scenario + '/dataset-ff-v05-param',
                        target_state=target_state, time=time, dtime=dtime, n_obs=n_obs, obs_position=obs_position,
                        obs_radius=obs_radius)


