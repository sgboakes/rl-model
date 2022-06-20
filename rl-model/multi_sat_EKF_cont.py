# -*- coding: utf-8 -*-
"""
Created on Thu Jan  20 13:29:40 2022

@author: sgboakes
"""

from __future__ import absolute_import, division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from abc import ABC
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg
import pandas as pd

import reverb
import tempfile

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tempdir = tempfile.gettempdir()


class PyEnvironment(object):

    def __init__(self):
        self._current_time_step = None

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, a):
        """Apply action and return new time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(a)
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def time_step_spec(self):
        """Return time_step_spec."""

    @abc.abstractmethod
    def observation_spec(self):
        """Return observation_spec."""

    @abc.abstractmethod
    def action_spec(self):
        """Return action_spec."""

    @abc.abstractmethod
    def _reset(self):
        """Return initial_time_step."""

    @abc.abstractmethod
    def _step(self, a):
        """Apply action and return new time_step."""


class SatEnv(py_environment.PyEnvironment, ABC):

    def __init__(self):
        super().__init__()
        global satAERMesT, satECIMesT
        self._num_look_spots = num_sats
        # Action size based on slew: 2deg/s
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-np.deg2rad(2*stepLength), maximum=np.deg2rad(2*stepLength), name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_sats,), dtype=np.float32, minimum=0, name='observation')  # distance
        self._episode_ended = False
        self._current_episode = 0
        self._max_episode_length = simLength
        # Field of view: pi/2
        self._sens_state = np.array((0, pi / 2), dtype=np.float32)  # Sensor state, az_mid, elev_mid
        global satAERMesT, satECIMesT
        satAERMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
        satECIMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # global satAERMesT, satECIMesT
        self._episode_ended = False
        self._current_episode = 0
        self._sens_state = np.array((0, pi / 2), dtype=np.float32)  # Sensor state, az_mid, elev_mid
        # import pdb; pdb.set_trace()
        global satAERMesT, satECIMesT
        satAERMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
        satECIMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
        return ts.restart(np.zeros(num_sats, dtype=np.float32))

    def _step(self, action):

        reward = 0
        global satAERMesT, satECIMesT
        distance = [None] * num_sats
        j = self._current_episode
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        '''
        Trying to point a telescope in the direction of satellites
        '''

        # Action: 0 = azimuth, 1 = elevation
        # LT Slew: 2deg/s
        if self._sens_state[0] + action[0] > pi:
            self._sens_state[0] = (self._sens_state[0] + action[0]) % pi
        elif self._sens_state[0] + action[0] < 0:
            self._sens_state[0] = pi - self._sens_state[0] + action[0]
        else:
            self._sens_state[0] += action[0]

        if (self._sens_state[1] + action[1]) > (pi/2):
            self._sens_state[1] = (pi/2) - ((self._sens_state[1] + action[1]) % pi/2)
        elif self._sens_state[1] + action[1] < 0:
            self._sens_state[1] = self._sens_state[1]
        else:
            self._sens_state[1] += action[1]

        sens_list[j, :] = self._sens_state

        for i in range(num_sats):
            c = chr(i + 97)
            if abs(satAER[c][0, j] - self._sens_state[0]) < 0.3 and abs(satAER[c][1, j] - self._sens_state[1]) < 0.3:
                reward += 1
                satAERMesT[c][:, j] = satAERMes[c][:, j]
                satECIMesT[c][:, j] = satECIMes[c][:, j]
            else:
                satAERMesT[c][:, j] = np.reshape(([[0.], [0.], [0.]]), (3,))
                satECIMesT[c][:, j] = np.reshape(([[0.], [0.], [0.]]), (3,))

            distance[i] = np.sqrt(abs(satAER[c][0, j] - self._sens_state[0]) ** 2 + abs(satAER[c][1, j] -
                                                                                        self._sens_state[1]) ** 2)

        distance = np.array(distance, dtype=np.float32)
        # print(distance)

        self._current_episode += 1

        if self._current_episode >= self._max_episode_length:
            # print('here')
            # import pdb; pdb.set_trace() # ALWAYS ON SAME LINE
            self._episode_ended = True
            return ts.termination(distance, reward)
        else:
            # print('here1')
            return ts.transition(distance, reward=reward, discount=1.0)

    # Make Heuristic decision maker here and compare to RL policy at end of training?
    # Maybe needs to go in environment class?
    # def heuristic(self, action):
    #     reward = 1
    #     self._current_episode += 1
    #     return ts.transition(distance, reward=reward, discount=1.0)


def generate_measurements():

    global satVisCheck
    rad_arr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    omega_arr = 1 / np.sqrt(rad_arr ** 3 / mu)
    theta_arr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    k_arr = np.ones((num_sats, 3), dtype='float64')
    k_arr[:, :] = 1 / np.sqrt(3)

    # Make data structures
    sat_eci = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_aer = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    sat_aer_check = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[rad_arr[i] * sin(omega_arr[i] * (j + 1) * stepLength)],
                          [0],
                          [rad_arr[i] * cos(omega_arr[i] * (j + 1) * stepLength)]], dtype='float64')

            sat_eci[c][:, j] = (v @ cos(theta_arr[i])) + (np.cross(k_arr[i, :].T, v.T) * sin(theta_arr[i])) + (
                               k_arr[i, :].T * np.dot(k_arr[i, :].T, v) * (1 - cos(theta_arr[i])))

            sat_aer[c][:, j:j+1] = Transformations.eci_to_aer(sat_eci[c][:, j], stepLength, j + 1, sensECEF, sensLLA[0],
                                                              sensLLA[1])

            if trans_earth:
                sat_aer_check[c][:, j:j+1] = sat_aer[c][:, j:j+1]

            if not trans_earth:
                if sat_aer[c][1, j] < 0:
                    sat_aer[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])
            elif trans_earth:
                if sat_aer_check[c][1, j] < 0:
                    sat_aer_check[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if (not trans_earth) & np.isnan(sat_aer[c]).all():
            print('Satellite {s} is not observable'.format(s=c))
            satVisCheck[c] = False
        elif trans_earth & np.isnan(sat_aer_check[c]).all():
            print('Satellite {s} is not observable'.format(s=c))
            satVisCheck[c] = False

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    ang_meas_dev, range_meas_dev = 1e-6, 20

    sat_aer_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        sat_aer_mes[c][0, :] = sat_aer[c][0, :] + (ang_meas_dev * np.random.randn(1, simLength))
        sat_aer_mes[c][1, :] = sat_aer[c][1, :] + (ang_meas_dev * np.random.randn(1, simLength))
        sat_aer_mes[c][2, :] = sat_aer[c][2, :] + (range_meas_dev * np.random.randn(1, simLength))

    sat_eci_mes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            sat_eci_mes[c][:, j:j + 1] = Transformations.aer_to_eci(sat_aer_mes[c][:, j], stepLength, j + 1, sensECEF,
                                                                    sensLLA[0], sensLLA[1])

    return sat_aer, sat_eci, sat_aer_mes, sat_eci_mes


def filtering(c, sim_length, sat_aer_mes, sat_eci_mes):

    sat_state = np.zeros((6, 1))
    cov_state = np.float64(1e12) * np.identity(6)

    sat_state[0:3] = np.reshape(satECIMes[c][:, 0], (3, 1))

    # Process noise
    std_ang = np.float64(1e-5)
    coef_a = np.float64(0.25 * stepLength ** 4.0 * std_ang ** 2.0)
    coef_b = np.float64(stepLength ** 2.0 * std_ang ** 2.0)
    coef_c = np.float64(0.5 * stepLength ** 3.0 * std_ang ** 2.0)

    proc_noise = np.array([[coef_a, 0, 0, coef_c, 0, 0],
                           [0, coef_a, 0, 0, coef_c, 0],
                           [0, 0, coef_a, 0, 0, coef_c],
                           [coef_c, 0, 0, coef_b, 0, 0],
                           [0, coef_c, 0, 0, coef_b, 0],
                           [0, 0, coef_c, 0, 0, coef_b]],
                          dtype='float64')

    ang_meas_dev, range_meas_dev = 1e-6, 20

    cov_aer = np.array([[(ang_meas_dev * 180 / pi) ** 2, 0, 0],
                        [0, (ang_meas_dev * 180 / pi) ** 2, 0],
                        [0, 0, range_meas_dev ** 2]],
                       dtype='float64')

    measure_matrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

    delta = 1e-6
    for j in range(sim_length):

        func_params = {
            "stepLength": stepLength,
            "count": j + 1,
            "sensECEF": sensECEF,
            "sensLLA[0]": sensLLA[0],
            "sensLLA[1]": sensLLA[1]}

        jacobian = Functions.jacobian_finder("aer_to_eci", np.reshape(sat_aer_mes[:, j], (3, 1)), func_params, delta)

        cov_eci = jacobian @ cov_aer @ jacobian.T

        state_trans_matrix = Functions.jacobian_finder("kepler", sat_state, [], delta)

        sat_state, cov_state = Filters.ekf(sat_state, cov_state, sat_eci_mes[:, j], state_trans_matrix,
                                           measure_matrix, cov_eci, proc_noise)

    return np.trace(cov_state)


if __name__ == "__main__":

    plt.close('all')
    np.random.seed(2)
    # ~~~~ Variables

    # Hyper-parameters ~~~~~~~~~~~~
    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    num_iterations = 100000  # @param {type:"integer"}

    initial_collect_steps = 10000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}

    batch_size = 1  # @param {type:"integer"}

    critic_learning_rate = 3e-4  # @param {type:"number"}
    actor_learning_rate = 3e-4  # @param {type:"number"}
    alpha_learning_rate = 3e-4  # @param {type:"number"}
    target_update_tau = 0.005  # @param {type:"number"}
    target_update_period = 1  # @param {type:"number"}
    gamma = 0.99  # @param {type:"number"}
    reward_scale_factor = 1.0  # @param {type:"number"}

    actor_fc_layer_params = (256, 256)  # size of nn?
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 5000  # @param {type:"integer"}

    num_eval_episodes = 20  # @param {type:"integer"}
    eval_interval = 10000  # @param {type:"integer"}

    policy_save_interval = 5000  # @param {type:"integer"}

    # ~~~~~~~~~~~~~~~~~~~~~~~

    sin = np.sin
    cos = np.cos
    pi = np.float64(np.pi)

    sensLat = np.float64(28.300697)
    sensLon = np.float64(-16.509675)
    sensAlt = np.float64(2390)
    sensLLA = np.array([[sensLat * pi / 180], [sensLon * pi / 180], [sensAlt]], dtype='float64')
    # sensLLA = np.array([[pi/2], [0], [1000]], dtype='float64')
    sensECEF = Transformations.lla_to_ecef(sensLLA)
    sensECEF.shape = (3, 1)

    # simLength = cfg.simLength
    simLength = 20
    stepLength = cfg.stepLength

    mu = cfg.mu

    trans_earth = True

    # ~~~~ Satellite Conversion

    # Define sat pos in ECI and convert to AER
    # radArr: radii for each sat metres
    # omegaArr: orbital rate for each sat rad/s
    # thetaArr: inclination angle for each sat rad
    # kArr: normal vector for each sat metres

    num_sats = 25
    # global satAER, satECI, satAERMes, satECIMes
    satVisCheck = {chr(i + 97): True for i in range(num_sats)}
    satAER, satECI, satAERMes, satECIMes = generate_measurements()
    satAERMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satECIMesT = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    # Field of view: pi/2
    sensState = np.array((0, pi/2), dtype=np.float32)  # Sensor state, az_mid, elev_mid
    sens_list = np.zeros((simLength, 2), dtype=np.float32)

    # tr_prior = {chr(i + 97): np.zeros((3, 3)) for i in range(num_sats)}
    # tr_posterior = {chr(i + 97): np.zeros((3, 3)) for i in range(num_sats)}
    # info_gain = [0.] * num_sats

    # ~~~~ Environment validation
    env = SatEnv()
    utils.validate_py_environment(env, episodes=5)

    # tf_env = tf_py_environment.TFPyEnvironment(env)
    # time_step = tf_env.reset()
    # rewards = []
    # steps = []
    # num_episodes = 5
    #
    # for _ in range(num_episodes):
    #     episode_reward = 0
    #     episode_steps = 0
    #     while not time_step.is_last():
    #         action = tf.random.uniform([2], -1, 1, dtype=tf.int32)
    #         time_step = tf_env.step(action)
    #         episode_steps += 1
    #         episode_reward += time_step.reward.numpy()
    #     rewards.append(episode_reward)
    #     steps.append(episode_steps)
    #     time_step = tf_env.reset()
    #
    # num_steps = np.sum(steps)
    # avg_length = np.mean(steps)
    # avg_reward = np.mean(rewards)
    #
    # print('num_episodes:', num_episodes, 'num_steps:', num_steps)
    # print('avg_length', avg_length, 'avg_reward:', avg_reward)

    env.reset()

    collect_env = env
    eval_env = env

    # ~~~~~~~~~~~~~~~ STRATEGY ~~~~~~~~~~~~~~~
    use_gpu = False
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

    # ~~~~~~~~~~~~~~~ AGENT ~~~~~~~~~~~~~~~

    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))

    with strategy.scope():
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
                tanh_normal_projection_network.TanhNormalProjectionNetwork))

    with strategy.scope():
        train_step = train_utils.create_train_step()

        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            train_step_counter=train_step)

        tf_agent.initialize()

    # ~~~~~~~~~~~~~~~ REPLAY BUFFER ~~~~~~~~~~~~~~~

    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)

    dataset = reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    # ~~~~~~~~~~~~~~~ POLICIES ~~~~~~~~~~~~~~~

    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())

    # ~~~~~~~~~~~~~~~ ACTORS ~~~~~~~~~~~~~~~

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        observers=[rb_observer])
    initial_collect_actor.run()

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])

    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'eval'),
    )

    rnd_actor = actor.Actor(
        eval_env,
        random_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=actor.eval_metrics(num_eval_episodes),
        summary_dir=os.path.join(tempdir, 'random'),
    )

    # ~~~~~~~~~~~~~~~ LEARNERS ~~~~~~~~~~~~~~~

    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers,
        strategy=strategy)

    # ~~~~~~~~~~~~~~~ METRICS AND EVALUATION ~~~~~~~~~~~~~~~

    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results


    def get_rnd_metrics():
        rnd_actor.run()
        results = {}
        for metric in rnd_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()


    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))


    log_eval_metrics(0, metrics)

    # ~~~~~~~~~~~~~~~ TRAINING AGENT ~~~~~~~~~~~~~~~

    covState = {chr(i + 97): np.zeros((6, 6)) for i in range(num_sats)}
    cov_init = 0
    cov_returns = {chr(i + 97): [] for i in range(num_sats)}
    cov_rnd_returns = {chr(i + 97): [] for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        covState[c] = 1e10 * np.identity(6)
        cov_init += np.trace(covState[c])
        cov_returns[c] = [cov_init]
        cov_rnd_returns[c] = [cov_init]

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]
    rnd_return = get_rnd_metrics()["AverageReturn"]
    rnd_returns = [rnd_return]

    satAERMesTF = {chr(i+97): np.zeros((3, simLength)) for i in range(num_sats)}
    satECIMesTF = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satAERMesTFR = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satECIMesTFR = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for _ in range(num_iterations):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        if log_interval and step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

        # Ignore these metrics - run episode on policy and save covs that way? Maybe reuse comp_avg_return from DDQN
        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])
            for i in range(num_sats):
                c = chr(i+97)
                if satVisCheck[c]:
                    # print('Measurements: {0}'.format(np.count_nonzero(satECIMesT[c])))
                    cov_returns[c].append(filtering(c, simLength, satAERMesT[c], satECIMesT[c]))
                    if step == num_iterations:
                        satAERMesTF[c] = satAERMesT[c]
                        satECIMesTF[c] = satECIMesT[c]
            rnd_metrics = get_rnd_metrics()
            log_eval_metrics(step, rnd_metrics)
            rnd_returns.append(rnd_metrics["AverageReturn"])
            for i in range(num_sats):
                c = chr(i+97)
                if satVisCheck[c]:
                    # print('Measurements: {0}'.format(np.count_nonzero(satECIMesT[c])))
                    cov_rnd_returns[c].append(filtering(c, simLength, satAERMesT[c], satECIMesT[c]))
                    if step == num_iterations:
                        satAERMesTFR[c] = satAERMesT[c]
                        satECIMesTFR[c] = satECIMesT[c]

    rb_observer.close()
    reverb_server.stop()

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.figure()
    plt.plot(iterations, returns, label='SAC Policy')
    plt.plot(iterations, rnd_returns, label='Random Policy')
    plt.legend()
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()

    plt.figure()
    for i in range(num_sats):
        c = chr(i + 97)
        if satVisCheck[c]:
            plt.plot(iterations, cov_returns[c])
    plt.yscale('log')
    plt.ylabel('Trace(P), Agent Policy')
    plt.xlabel('Iterations')
    plt.show()

    plt.figure()
    for i in range(num_sats):
        c = chr(i + 97)
        if satVisCheck[c]:
            plt.plot(iterations, cov_rnd_returns[c])
    plt.yscale('log')
    plt.ylabel('Trace(P), Random Policy')
    plt.xlabel('Iterations')
    plt.show()

    cov_final = {chr(i + 97): [] for i in range(num_sats)}
    cov_rnd_final = {chr(i + 97): [] for i in range(num_sats)}

    # Random Cov Results
    for i in range(num_sats):
        c = chr(i + 97)
        cov_final[c] = []
        sat_state = np.zeros((6, 1))
        cov_state = np.float64(1e12) * np.identity(6)

        sat_state[0:3] = np.reshape(satECIMes[c][:, 0], (3, 1))

        # Process noise
        std_ang = np.float64(1e-5)
        coef_a = np.float64(0.25 * stepLength ** 4.0 * std_ang ** 2.0)
        coef_b = np.float64(stepLength ** 2.0 * std_ang ** 2.0)
        coef_c = np.float64(0.5 * stepLength ** 3.0 * std_ang ** 2.0)

        proc_noise = np.array([[coef_a, 0, 0, coef_c, 0, 0],
                               [0, coef_a, 0, 0, coef_c, 0],
                               [0, 0, coef_a, 0, 0, coef_c],
                               [coef_c, 0, 0, coef_b, 0, 0],
                               [0, coef_c, 0, 0, coef_b, 0],
                               [0, 0, coef_c, 0, 0, coef_b]],
                              dtype='float64')

        ang_meas_dev, range_meas_dev = 1e-6, 20

        cov_aer = np.array([[(ang_meas_dev * 180 / pi) ** 2, 0, 0],
                            [0, (ang_meas_dev * 180 / pi) ** 2, 0],
                            [0, 0, range_meas_dev ** 2]],
                           dtype='float64')

        measure_matrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

        delta = 1e-6
        for j in range(simLength):
            func_params = {
                "stepLength": stepLength,
                "count": j + 1,
                "sensECEF": sensECEF,
                "sensLLA[0]": sensLLA[0],
                "sensLLA[1]": sensLLA[1]}

            jacobian = Functions.jacobian_finder("aer_to_eci", np.reshape(satAERMesTFR[c][:, j], (3, 1)),
                                                 func_params, delta)

            cov_eci = jacobian @ cov_aer @ jacobian.T

            state_trans_matrix = Functions.jacobian_finder("kepler", sat_state, [], delta)

            sat_state, cov_state = Filters.ekf(sat_state, cov_state, satECIMesTFR[c][:, j], state_trans_matrix,
                                                   measure_matrix, cov_eci, proc_noise)

            cov_rnd_final[c].append(np.trace(cov_state))

    # Agent Cov Results
    for i in range(num_sats):
        c = chr(i + 97)
        cov_final[c] = []
        sat_state = np.zeros((6, 1))
        cov_state = np.float64(1e12) * np.identity(6)

        sat_state[0:3] = np.reshape(satECIMes[c][:, 0], (3, 1))

        # Process noise
        std_ang = np.float64(1e-5)
        coef_a = np.float64(0.25 * stepLength ** 4.0 * std_ang ** 2.0)
        coef_b = np.float64(stepLength ** 2.0 * std_ang ** 2.0)
        coef_c = np.float64(0.5 * stepLength ** 3.0 * std_ang ** 2.0)

        proc_noise = np.array([[coef_a, 0, 0, coef_c, 0, 0],
                               [0, coef_a, 0, 0, coef_c, 0],
                               [0, 0, coef_a, 0, 0, coef_c],
                               [coef_c, 0, 0, coef_b, 0, 0],
                               [0, coef_c, 0, 0, coef_b, 0],
                               [0, 0, coef_c, 0, 0, coef_b]],
                              dtype='float64')

        ang_meas_dev, range_meas_dev = 1e-6, 20

        cov_aer = np.array([[(ang_meas_dev * 180 / pi) ** 2, 0, 0],
                            [0, (ang_meas_dev * 180 / pi) ** 2, 0],
                            [0, 0, range_meas_dev ** 2]],
                           dtype='float64')

        measure_matrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

        delta = 1e-6
        for j in range(simLength):
            func_params = {
                "stepLength": stepLength,
                "count": j + 1,
                "sensECEF": sensECEF,
                "sensLLA[0]": sensLLA[0],
                "sensLLA[1]": sensLLA[1]}

            jacobian = Functions.jacobian_finder("aer_to_eci", np.reshape(satAERMesTF[c][:, j], (3, 1)),
                                                 func_params, delta)

            cov_eci = jacobian @ cov_aer @ jacobian.T

            state_trans_matrix = Functions.jacobian_finder("kepler", sat_state, [], delta)

            sat_state, cov_state = Filters.ekf(sat_state, cov_state, satECIMesTF[c][:, j],
                                                   state_trans_matrix,
                                                   measure_matrix, cov_eci, proc_noise)

            cov_final[c].append(np.trace(cov_state))

    save_check = input("Do you want to save output to csv file? (y/n) \n")
    if save_check == ('y' or 'Y'):

        # df_reward = pd.DataFrame()
        # df_reward['Iterations'] = iterations
        # df_reward['Returns'] = returns
        # df_reward['Rnd_Returns'] = rnd_returns
        #
        # df_reward.to_csv('Reward-Returns-5.csv')
        #
        # df_cov = pd.DataFrame()
        # df_cov['Iterations'] = iterations
        # for i in range(num_sats):
        #     c = chr(i+97)
        #     if satVisCheck[c]:
        #         df_cov['Cov_{c}'.format(c=c)] = cov_returns[c]
        #         df_cov['Cov_rnd_{c}'.format(c=c)] = cov_rnd_returns[c]
        #
        # df_cov.to_csv('Covariance-Returns-5.csv')

        df_cov_fin = pd.DataFrame()
        df_cov_fin['Steps'] = range(0, simLength)
        for i in range(num_sats):
            c = chr(i+97)
            if satVisCheck[c]:
                df_cov_fin['Cov_{c}'.format(c=c)] = cov_final[c]
                df_cov_fin['Cov_Rnd_{c}'.format(c=c)] = cov_rnd_final[c]

        df_cov_fin.to_csv('Covariance-Fin-Returns-5.csv')
