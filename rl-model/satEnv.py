# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:14:48 2021

@author: sgboakes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


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


class satEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=10, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(10,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False
        self._episode_duration = 0
        import pdb; pdb.set_trace()  # debugging
        self._max_episode_length = 20

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        # Initialise satellite to go 1 through 10 or backwards
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIX

        self._episode_duration += 1

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == self._state:
            reward = 1
        else:
            reward = 0

        self._state = (self._state + 1) % 10

        state = [0] * 10
        state[self._state] = 1
        if self._episode_duration >= self._max_episode_length:
            return ts.termination(np.array([state], dtype=np.int32), reward)

        return ts.transition(
            np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
