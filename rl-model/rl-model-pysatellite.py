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

import pysatellite.Functions
import tensorflow as tf
import numpy as np
from numpy import int32, float32

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
# import base64
# import imageio
# import IPython
import matplotlib
import matplotlib.pyplot as plt
# import PIL.Image
# import pyvirtualdisplay
import reverb
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from pysatellite import Transformations, Functions, Filters
import pysatellite.config as cfg

if __name__ == "__main__":

    plt.close('all')
    np.random.seed(4)
    # ~~~~ Variables

    # Hyper-parameters
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

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
    sensECEF = Transformations.LLAtoECEF(sensLLA)
    sensECEF.shape = (3, 1)

    simLength = cfg.simLength
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

    num_sats = 6
    radArr = 7e6 * np.ones((num_sats, 1), dtype='float64')
    omegaArr = 1 / np.sqrt(radArr ** 3 / mu)
    thetaArr = np.array((2 * pi * np.random.rand(num_sats, 1)), dtype='float64')
    kArr = np.ones((num_sats, 3), dtype='float64')
    kArr[:, :] = 1 / np.sqrt(3)

    # Make data structures
    satECI = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    satAER = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}

    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            v = np.array([[radArr[i] * sin(omegaArr[i] * (j + 1) * stepLength)],
                          [0],
                          [radArr[i] * cos(omegaArr[i] * (j + 1) * stepLength)]], dtype='float64')

            satECI[c][:, j] = (v @ cos(thetaArr[i])) + (np.cross(kArr[i, :].T, v.T) * sin(thetaArr[i])) + (
                    kArr[i, :].T * np.dot(kArr[i, :].T, v) * (1 - cos(thetaArr[i])))

            satAER[c][:, j:j + 1] = Transformations.ECItoAER(satECI[c][:, j], stepLength, j + 1, sensECEF, sensLLA[0],
                                                             sensLLA[1])

            if not trans_earth:
                if satAER[c][1, j] < 0:
                    satAER[c][:, j:j + 1] = np.array([[np.nan], [np.nan], [np.nan]])

        if np.isnan(satAER[c]).all():
            print('Satellite {s} is not observable'.format(s=c))

    # Add small deviations for measurements
    # Using calculated max measurement deviations for LT:
    # Based on 0.15"/pixel, sat size = 2m, max range = 1.38e7
    # sigma = 1/2 * 0.15" for it to be definitely on that pixel
    # Add angle devs to Az/Elev, and range devs to Range

    angMeasDev, rangeMeasDev = 1e-6, 20

    satAERMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        satAERMes[c][0, :] = satAER[c][0, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][1, :] = satAER[c][1, :] + (angMeasDev * np.random.randn(1, simLength))
        satAERMes[c][2, :] = satAER[c][2, :] + (rangeMeasDev * np.random.randn(1, simLength))

    satECIMes = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            satECIMes[c][:, j:j + 1] = Transformations.AERtoECI(satAERMes[c][:, j], stepLength, j + 1, sensECEF,
                                                                sensLLA[0], sensLLA[1])

    satState = {chr(i + 97): np.zeros((6, 1)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        for j in range(simLength):
            if np.all(np.isnan(satECIMes[c][:, j])):
                continue
            else:
                satState[c][0:3] = np.reshape(satECIMes[c][:, j], (3, 1))
                break

    # Process noise
    stdAng = np.float64(1e-5)
    coefA = np.float64(0.25 * stepLength ** 4.0 * stdAng ** 2.0)
    coefB = np.float64(stepLength ** 2.0 * stdAng ** 2.0)
    coefC = np.float64(0.5 * stepLength ** 3.0 * stdAng ** 2.0)

    procNoise = np.array([[coefA, 0, 0, coefC, 0, 0],
                          [0, coefA, 0, 0, coefC, 0],
                          [0, 0, coefA, 0, 0, coefC],
                          [coefC, 0, 0, coefB, 0, 0],
                          [0, coefC, 0, 0, coefB, 0],
                          [0, 0, coefC, 0, 0, coefB]],
                         dtype='float64')

    covState = {chr(i + 97): np.zeros((6, 6)) for i in range(num_sats)}
    for i in range(num_sats):
        c = chr(i + 97)
        covState[c] = np.float64(1e10) * np.identity(6)

    covAER = np.array([[(angMeasDev * 180 / pi) ** 2, 0, 0],
                       [0, (angMeasDev * 180 / pi) ** 2, 0],
                       [0, 0, rangeMeasDev ** 2]],
                      dtype='float64')

    measureMatrix = np.append(np.identity(3), np.zeros((3, 3)), axis=1)

    # ~~~~~~ CHECK
    totalStates = {chr(i + 97): np.zeros((6, simLength)) for i in range(num_sats)}
    diffState = {chr(i + 97): np.zeros((3, simLength)) for i in range(num_sats)}
    err_X_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Y_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}
    err_Z_ECI = {chr(i + 97): np.zeros(simLength) for i in range(num_sats)}

    # tr_prior = {chr(i + 97): np.zeros((3, 3)) for i in range(num_sats)}
    # tr_posterior = {chr(i + 97): np.zeros((3, 3)) for i in range(num_sats)}
    # info_gain = [0.] * num_sats
    # ~~~~~ Using EKF

    delta = 1e-6
    # for i in range(num_sats):
    #     c = chr(i + 97)
    #     mesCheck = False
    #     for j in range(simLength):
    #         while not mesCheck:
    #             if np.all(np.isnan(satECIMes[c][:, j])):
    #                 break
    #             else:
    #                 mesCheck = True
    #                 break
    #
    #         if not mesCheck:
    #             continue
    #
    #         func_params = {
    #             "stepLength": stepLength,
    #             "count": j + 1,
    #             "sensECEF": sensECEF,
    #             "sensLLA[0]": sensLLA[0],
    #             "sensLLA[1]": sensLLA[1]
    #         }
    #
    #         jacobian = Functions.jacobian_finder("AERtoECI", np.reshape(satAERMes[c][:, j], (3, 1)), func_params,
    #                                              delta)
    #
    #         # covECI = np.matmul(np.matmul(jacobian, covAER), jacobian.T)
    #         covECI = jacobian @ covAER @ jacobian.T
    #
    #         stateTransMatrix = Functions.jacobian_finder("kepler", satState[c], [], delta)
    #
    #         satState[c], covState[c] = Filters.EKF_ECI(satState[c], covState[c], satECIMes[c][:, j], stateTransMatrix,
    #                                                    measureMatrix, covECI, procNoise)
    #
    #         totalStates[c][:, j] = np.reshape(satState[c], 6)
    #         err_X_ECI[c][j] = (np.sqrt(np.abs(covState[c][0, 0])))
    #         err_Y_ECI[c][j] = (np.sqrt(np.abs(covState[c][1, 1])))
    #         err_Z_ECI[c][j] = (np.sqrt(np.abs(covState[c][2, 2])))
    #         diffState[c][:, j] = totalStates[c][0:3, j] - satECI[c][:, j]
    #         print(satState[c])


class PyEnvironment(object):

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Apply action and return new time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
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
    def _step(self, action):
        """Apply action and return new time_step."""


class SatEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._num_look_spots = num_sats
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._num_look_spots - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_sats,), dtype=np.float32, minimum=0, name='observation')
        self._episode_ended = False
        self._current_episode = 0
        self._max_episode_length = simLength
        self._cov_state = {chr(i + 97): np.float64(1e6) * np.identity(6) for i in range(num_sats)}

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # self._state = 0
        self._episode_ended = False
        self._current_episode = 0
        self._cov_state = {chr(i + 97): np.float64(1e6) * np.identity(6) for i in range(num_sats)}
        # import pdb; pdb.set_trace()
        return ts.restart(np.zeros(num_sats, dtype=np.float32))

    def _step(self, action):

        reward = 0
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        '''
        Want to compare the prior covariance for each satellite to the posterior, and if action was the largest
        information gain (largest difference between prior and posterior), then gain reward. Information gain quantified
        in trace of covariance. int(step/eval_interval)-1
        '''

        tr_prior = [0.] * num_sats  # Trace of prior covariance
        tr_posterior = [0.] * num_sats  # Trace of posterior covariance
        info_gain = [0.] * num_sats  # Information gain
        j = self._current_episode

        # ~~~~~ Using EKF
        for i in range(num_sats):
            c = chr(i + 97)
            # Get prior information
            tr_prior[i] = np.trace(self._cov_state[c])

            if i != action:
                satECIMes[c][:, j] = np.reshape(([[np.nan], [np.nan], [np.nan]]), (3,))

            func_params = {
                "stepLength": stepLength,
                "count": j + 1,
                "sensECEF": sensECEF,
                "sensLLA[0]": sensLLA[0],
                "sensLLA[1]": sensLLA[1]}

            jacobian = Functions.jacobian_finder("AERtoECI", np.reshape(satAERMes[c][:, j], (3, 1)),
                                                 func_params, delta)

            covECI = jacobian @ covAER @ jacobian.T

            stateTransMatrix = Functions.jacobian_finder("kepler", satState[c], [], delta)

            satState[c], self._cov_state[c] = Filters.EKF_ECI(satState[c], self._cov_state[c], satECIMes[c][:, j],
                                                              stateTransMatrix, measureMatrix, covECI, procNoise)

            tr_posterior[i] = np.trace(self._cov_state[c])
            if np.isnan(tr_posterior[i]):
                tr_posterior[i] = 0
            info_gain[i] = tr_posterior[i] - tr_prior[i]
            info_gain[i] = np.linalg.norm(tr_posterior[i])

            if info_gain[i] < 0:
                info_gain[i] = 0
            else:
                info_gain[i] = np.sqrt(info_gain[i])

        # print('simlength {s} current ep {e}'.format(s=simLength,e=self._current_episode))

        sorted_info = sorted(info_gain)
        # if action == info_gain.index(max(sorted_info)):
        #     reward = 1
        # elif action == info_gain.index(sorted_info[0]):
        #     reward = -1
        # else:
        #     reward = 0

        if action == info_gain.index(max(sorted_info)):
            reward = num_sats * 4
        else:
            for i in sorted_info:
                if action == info_gain.index(i):
                    reward = sorted_info.index(i)


        # print(reward)
        self._current_episode += 1
        info_gain = np.array(info_gain, dtype=np.float32)
        info_gain = np.nan_to_num(info_gain, nan=1)
        info_gain[:] = info_gain[:] / np.sum(info_gain)
        tr_posterior = np.array(np.sqrt(tr_posterior), dtype=np.float32)

        tr_posterior = np.nan_to_num(tr_posterior, nan=0)
        # print(tr_posterior)
        tr_posterior[:] = tr_posterior[:] / np.sum(tr_posterior)

        if self._current_episode >= self._max_episode_length:
            # print('here')
            # import pdb; pdb.set_trace() # ALWAYS ON SAME LINE
            self._episode_ended = True
            return ts.termination(info_gain, reward)
        else:
            # print('here1')
            return ts.transition(info_gain, reward=reward, discount=1.0)


env = SatEnv()
utils.validate_py_environment(env, episodes=5)

tf_env = tf_py_environment.TFPyEnvironment(env)
time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 8

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


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


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

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
def compute_avg_return(environment, policy, num_episodes=10):
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

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
rnd_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)
returns = [avg_return]
rnd_returns = [rnd_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

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
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        rnd_return = compute_avg_return(eval_env, random_policy, num_eval_episodes)

        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        rnd_returns.append(rnd_return)

# @test {"skip": true}

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns, label='DQN Policy')
plt.plot(iterations, rnd_returns, label='Random Policy')
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.legend()
plt.show()
# plt.ylim(top=250)
