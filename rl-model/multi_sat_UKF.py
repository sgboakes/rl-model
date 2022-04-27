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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg
import pandas as pd
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints

import reverb
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


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
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')  # can look up, down, left, right, or none
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_sats,), dtype=np.float32, minimum=0, name='observation')
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

        # Action: 0=up, 1=down, 2=left, 3=right, 4=nothing
        # LT Slew: 2deg/s
        slew = np.deg2rad(2 * stepLength)
        if action == 0:
            if not (self._sens_state[1] + slew) >= pi:
                self._sens_state[1] += slew  # Radians
        elif action == 1:
            if not (self._sens_state[1] - slew) <= 0:
                self._sens_state[1] -= slew  # Radians
        elif action == 2:
            if (self._sens_state[0] + slew) >= (2 * pi):
                self._sens_state[0] = abs(self._sens_state[0] - (2 * pi))  # Radians
            else:
                self._sens_state[0] += slew
        elif action == 3:
            if (self._sens_state[0] - slew) <= 0:
                self._sens_state[0] = (2 * pi) - self._sens_state[0]  # Radians
            else:
                self._sens_state[0] -= slew
        elif action == 4:
            self._sens_state = self._sens_state
        else:
            raise ValueError('`action` should be between 0 and 4.')

        sens_list[j, :] = self._sens_state

        for i in range(num_sats):
            c = chr(i + 97)
            if abs(satAER[c][0, j] - self._sens_state[0]) < 0.5 and abs(satAER[c][1, j] - self._sens_state[1]) < 0.5:
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


def compute_avg_return(environment, policy, num_episodes=10):
    time_step = None
    episode_return = None
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def dense_layer(num_units):
    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                            scale=2.0, mode='fan_in', distribution='truncated_normal'))


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
    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)
    kf = UKF(dim_x=6, dim_z=3, dt=stepLength, fx=Functions.kepler, hx=Functions.h_x, points=points)

    kf.x = np.zeros((6, 1))
    kf.P = np.float64(1e12) * np.identity(6)

    kf.x[0:3] = np.reshape(satECI[c][:, 0], (3, 1))
    kf.x = kf.x.flat

    # Process noise
    std_ang = np.float64(1e-5)
    coef_a = np.float64(0.25 * stepLength ** 4.0 * std_ang ** 2.0)
    coef_b = np.float64(stepLength ** 2.0 * std_ang ** 2.0)
    coef_c = np.float64(0.5 * stepLength ** 3.0 * std_ang ** 2.0)

    kf.Q = np.array([[coef_a, 0, 0, coef_c, 0, 0],
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

    delta = np.float64(1e-6)
    for j in range(sim_length):

        func_params = {
            "stepLength": stepLength,
            "count": j + 1,
            "sensECEF": sensECEF,
            "sensLLA[0]": sensLLA[0],
            "sensLLA[1]": sensLLA[1]}

        jacobian = Functions.jacobian_finder("aer_to_eci", np.reshape(sat_aer_mes[:, j], (3, 1)), func_params, delta)

        kf.R = jacobian @ cov_aer @ jacobian.T

        # import pdb; pdb.set_trace()
        kf.predict()
        if not np.any(np.isnan(sat_eci_mes[:, j])) and np.any(sat_eci_mes[:, j]):
            kf.update(sat_eci_mes[:, j])

    return np.trace(kf.P)


if __name__ == "__main__":

    plt.close('all')
    np.random.seed(2)
    # ~~~~ Variables

    # Hyper-parameters
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-4  # @param {type:"number"}
    log_interval = 500  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

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

    env = SatEnv()
    utils.validate_py_environment(env, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(env)
    time_step = tf_env.reset()
    rewards = []
    steps = []
    num_episodes = 5

    for _ in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()
        rewards.append(episode_reward)
        steps.append(episode_steps)
        time_step = tf_env.reset()

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)

    print('num_episodes:', num_episodes, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)

    env.reset()

    train_py_env = env
    eval_py_env = env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    # @test {"skip": true}
    # See also the metrics module for standard implementations of different metrics.
    # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
          agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      replay_buffer.py_client,
      table_name,
      sequence_length=2)

    # agent.collect_data_spec
    # agent.collect_data_spec._fields

    # @test {"skip": true}
    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
          random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # #@test {"skip": true}
    # try:
    #   % % time
    # except:
    #   pass

    # (Optional) Optimize by wrapping some code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    rnd_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
    returns = [avg_return]
    rnd_returns = [rnd_return]

    covState = {chr(i + 97): np.zeros((6, 6)) for i in range(num_sats)}
    cov_init = 0
    cov_returns = {chr(i+97): [] for i in range(num_sats)}
    cov_rnd_returns = {chr(i+97): [] for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i+97)
        covState[c] = 1e10 * np.identity(6)
        cov_init += np.trace(covState[c])
        cov_returns[c] = [cov_init]
        cov_rnd_returns[c] = [cov_init]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    satAERMesTF = {chr(i+97): np.zeros((3, simLength)) for i in range(num_sats)}
    satECIMesTF = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satAERMesTFR = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satECIMesTFR = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for _ in range(num_iterations):
        # print('Iteration: {i}'.format(i=_))
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        # import pdb; pdb.set_trace()
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            # Random Policy Output
            rnd_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
            rnd_returns.append(rnd_return)
            for i in range(num_sats):
                c = chr(i+97)
                if satVisCheck[c]:
                    # print('Measurements: {0}'.format(np.count_nonzero(satECIMesT[c])))
                    cov_rnd_returns[c].append(filtering(c, simLength, satAERMesT[c], satECIMesT[c]))
                    if step == 20000:
                        satAERMesTFR[c] = satAERMesT[c]
                        satECIMesTFR[c] = satECIMesT[c]

            # Agent Policy Output
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            returns.append(avg_return)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            for i in range(num_sats):
                c = chr(i+97)
                if satVisCheck[c]:
                    # print('Measurements: {0}'.format(np.count_nonzero(satECIMesT[c])))
                    cov_returns[c].append(filtering(c, simLength, satAERMesT[c], satECIMesT[c]))
                    if step == 20000:
                        satAERMesTF[c] = satAERMesT[c]
                        satECIMesTF[c] = satECIMesT[c]

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.figure()
    plt.plot(iterations, returns, label='DQN Policy')
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
