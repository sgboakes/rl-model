from __future__ import absolute_import, division, print_function

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import reverb
import os
import tempfile

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.train import actor, learner, triggers
from tf_agents.train.utils import spec_utils, strategy_utils, train_utils

import abc
import tensorflow as tf


from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tempdir = tempfile.gettempdir()

# ~~~~~~~~~~~~~~~ HYPER-PARAMETERS ~~~~~~~~~~~~~~~

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 10000  # @param {type:"integer"}

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

# ~~~~~~~~~~~~~~~ ENVIRONMENT ~~~~~~~~~~~~~~~


class PyEnvironment(object):

    def __init__(self):
        self._current_time_step = None

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


class SatEnv(py_environment.PyEnvironment, ABC):

    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=0, name='observation')
        self._state = 0.
        self._episode_ended = False
        self._episode_duration = 0
        self._max_episode_length = 20

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # self._state = 0.
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):

        reward = 0
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._episode_duration += 1

        # Make sure episodes don't go on forever.
        if abs(action - self._state) < 0.1:
            reward = 1

        self._state = (self._state + .1) % 1.
        # print(self._state)
        # print("reward = ", reward)

        if self._episode_duration >= self._max_episode_length:
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32), reward=reward, discount=1.0)


env = SatEnv()
# import pdb; pdb.set_trace()
utils.validate_py_environment(env, episodes=10)

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

collect_env = env
eval_env = env

# collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

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


metrics = get_eval_metrics()


def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))


log_eval_metrics(0, metrics)


# ~~~~~~~~~~~~~~~ TRAINING AGENT ~~~~~~~~~~~~~~~

# try:
#     %%time
# except:
#     pass

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for _ in range(num_iterations):
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=1)

    # Evaluating.
    step = agent_learner.train_step_numpy

    if eval_interval and step % eval_interval == 0:
        metrics = get_eval_metrics()
        log_eval_metrics(step, metrics)
        returns.append(metrics["AverageReturn"])

    if log_interval and step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

# ~~~~~~~~~~~~~~~ VISUALISATION ~~~~~~~~~~~~~~~

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
