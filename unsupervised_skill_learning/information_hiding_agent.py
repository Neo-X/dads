# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TF-Agents Class for DADS. Builds on top of the SAC agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import object_identity
from tf_agents.agents import tf_agent
from tf_agents.agents.sac.sac_agent import SacLossInfo
from tf_agents.policies import actor_policy

import skill_dynamics
import observation_processor
import wrapped_policy

nest = tf.nest


class InformationHidingAgent(sac_agent.SacAgent):

  def __init__(self,
               save_directory,
               skill_dynamics_observation_size,
               observation_modify_fn=None,
               restrict_input_size=0,
               latent_size=2,
               latent_prior='cont_uniform',
               prior_samples=100,
               fc_layer_params=(256, 256),
               normalize_observations=True,
               network_type='default',
               num_mixture_components=4,
               fix_variance=True,
               skill_dynamics_learning_rate=3e-4,
               reweigh_batches=False,
               agent_graph=None,
               skill_dynamics_graph=None,
               information_hiding_weight=0.001,
               *sac_args,
               **sac_kwargs):
    self._skill_dynamics_learning_rate = skill_dynamics_learning_rate
    self._latent_size = latent_size
    self._latent_prior = latent_prior
    self._prior_samples = prior_samples
    self._save_directory = save_directory
    self._restrict_input_size = restrict_input_size
    self._process_observation = observation_modify_fn
    self._information_hiding_weight = information_hiding_weight

    if agent_graph is None:
      self._graph = tf.compat.v1.get_default_graph()
    else:
      self._graph = agent_graph

    if skill_dynamics_graph is None:
      skill_dynamics_graph = self._graph

    # instantiate the skill dynamics
    self._skill_dynamics = skill_dynamics.SkillDynamics(
        observation_size=skill_dynamics_observation_size,
        action_size=self._latent_size,
        restrict_observation=self._restrict_input_size,
        normalize_observations=normalize_observations,
        fc_layer_params=fc_layer_params,
        network_type=network_type,
        num_components=num_mixture_components,
        fix_variance=fix_variance,
        reweigh_batches=reweigh_batches,
        graph=skill_dynamics_graph)

    # instantiate the observation processor
    self._observation_processor = observation_processor.ObservationProcessor(
        observation_size=skill_dynamics_observation_size,
        restrict_observation=self._restrict_input_size,
        normalize_observations=normalize_observations,
        fc_layer_params=fc_layer_params,
        network_type=network_type,
        num_components=num_mixture_components,
        fix_variance=fix_variance,
        reweigh_batches=reweigh_batches,
        graph=skill_dynamics_graph)

    class WrappedPolicy(actor_policy.ActorPolicy):

        def __init__(this,
                     *args,
                     **kwargs):
            super(WrappedPolicy, this).__init__(*args, **kwargs)
            this._processor = self.observation_processor

        def _apply_actor_network(this, observation, step_type, policy_state,
                                 mask=None):
            this._processor.build_graph(observation)
            return super(WrappedPolicy, this)._apply_actor_network(
                this._processor.mean, step_type, policy_state, mask=mask)

    super(InformationHidingAgent, self).__init__(
        *sac_args,
        actor_policy_ctor=wrapped_policy.WrappedPolicy,
        **sac_kwargs)
    self._placeholders_in_place = False

  def compute_dads_reward(self, input_obs, cur_skill, target_obs):
    if self._process_observation is not None:
      input_obs, target_obs = self._process_observation(
          input_obs), self._process_observation(target_obs)

    num_reps = self._prior_samples if self._prior_samples > 0 else self._latent_size - 1
    input_obs_altz = np.concatenate([input_obs] * num_reps, axis=0)
    target_obs_altz = np.concatenate([target_obs] * num_reps, axis=0)

    # for marginalization of the denominator
    if self._latent_prior == 'discrete_uniform' and not self._prior_samples:
      alt_skill = np.concatenate(
          [np.roll(cur_skill, i, axis=1) for i in range(1, num_reps + 1)],
          axis=0)
    elif self._latent_prior == 'discrete_uniform':
      alt_skill = np.random.multinomial(
          1, [1. / self._latent_size] * self._latent_size,
          size=input_obs_altz.shape[0])
    elif self._latent_prior == 'gaussian':
      alt_skill = np.random.multivariate_normal(
          np.zeros(self._latent_size),
          np.eye(self._latent_size),
          size=input_obs_altz.shape[0])
    elif self._latent_prior == 'cont_uniform':
      alt_skill = np.random.uniform(
          low=-1.0, high=1.0, size=(input_obs_altz.shape[0], self._latent_size))

    logp = self._skill_dynamics.get_log_prob(input_obs, cur_skill, target_obs)

    # denominator may require more memory than that of a GPU, break computation
    split_group = 20 * 4000
    if input_obs_altz.shape[0] <= split_group:
      logp_altz = self._skill_dynamics.get_log_prob(input_obs_altz, alt_skill,
                                                    target_obs_altz)
    else:
      logp_altz = []
      for split_idx in range(input_obs_altz.shape[0] // split_group):
        start_split = split_idx * split_group
        end_split = (split_idx + 1) * split_group
        logp_altz.append(
            self._skill_dynamics.get_log_prob(
                input_obs_altz[start_split:end_split],
                alt_skill[start_split:end_split],
                target_obs_altz[start_split:end_split]))
      if input_obs_altz.shape[0] % split_group:
        start_split = input_obs_altz.shape[0] % split_group
        logp_altz.append(
            self._skill_dynamics.get_log_prob(input_obs_altz[-start_split:],
                                              alt_skill[-start_split:],
                                              target_obs_altz[-start_split:]))
      logp_altz = np.concatenate(logp_altz)
    logp_altz = np.array(np.array_split(logp_altz, num_reps))

    # final DADS reward
    intrinsic_reward = np.log(num_reps + 1) - np.log(1 + np.exp(
        np.clip(logp_altz - logp.reshape(1, -1), -50, 50)).sum(axis=0))

    return intrinsic_reward, {'logp': logp, 'logp_altz': logp_altz.flatten()}

  def get_experience_placeholder(self):
    self._placeholders_in_place = True
    self._placeholders = []
    for item in nest.flatten(self.collect_data_spec):
      self._placeholders += [
          tf.compat.v1.placeholder(
              item.dtype,
              shape=(None, 2) if len(item.shape) == 0 else
              (None, 2, item.shape[-1]),
              name=item.name)
      ]
    self._policy_experience_ph = nest.pack_sequence_as(self.collect_data_spec,
                                                       self._placeholders)
    return self._policy_experience_ph

  def build_agent_graph(self):
    with self._graph.as_default():
      self.get_experience_placeholder()
      self.agent_train_op = self.train(self._policy_experience_ph)
      self.summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
      return self.agent_train_op

  def build_skill_dynamics_graph(self):
    self._skill_dynamics.make_placeholders()
    self._skill_dynamics.build_graph()
    self._skill_dynamics.increase_prob_op(
        learning_rate=self._skill_dynamics_learning_rate)

  def build_observation_processor_graph(self):
    self._observation_processor.make_placeholders()
    self._observation_processor.build_graph()

  def create_savers(self):
    self._skill_dynamics.create_saver(
        save_prefix=os.path.join(self._save_directory, 'dynamics'))
    self._observation_processor.create_saver(
        save_prefix=os.path.join(self._save_directory, 'processor'))

  def set_sessions(self, initialize_or_restore_skill_dynamics, session=None):
    if session is not None:
      self._session = session
    else:
      self._session = tf.compat.v1.Session(graph=self._graph)
    self._skill_dynamics.set_session(
        initialize_or_restore_variables=initialize_or_restore_skill_dynamics,
        session=session)
    self._observation_processor.set_session(
        initialize_or_restore_variables=initialize_or_restore_skill_dynamics,
        session=session)

  def save_variables(self, global_step):
    self._skill_dynamics.save_variables(global_step=global_step)
    self._observation_processor.save_variables(global_step=global_step)

  def _get_dict(self, trajectories, batch_size=-1):
    tf.nest.assert_same_structure(self.collect_data_spec, trajectories)
    if batch_size > 0:
      shuffled_batch = np.random.permutation(
          trajectories.observation.shape[0])[:batch_size]
    else:
      shuffled_batch = np.arange(trajectories.observation.shape[0])

    return_dict = {}

    for placeholder, val in zip(self._placeholders, nest.flatten(trajectories)):
      return_dict[placeholder] = val[shuffled_batch]

    return return_dict

  def train_loop(self,
                 trajectories,
                 recompute_reward=False,
                 batch_size=-1,
                 num_steps=1):
    if not self._placeholders_in_place:
      return

    if recompute_reward:
      input_obs = trajectories.observation[:, 0, :-self._latent_size]
      cur_skill = trajectories.observation[:, 0, -self._latent_size:]
      target_obs = trajectories.observation[:, 1, :-self._latent_size]
      new_reward, info = self.compute_dads_reward(input_obs, cur_skill,
                                                  target_obs)
      trajectories = trajectories._replace(
          reward=np.concatenate(
              [np.expand_dims(new_reward, axis=1), trajectories.reward[:, 1:]],
              axis=1))

    # TODO(architsh):all agent specs should be the same as env specs, shift preprocessing to actor/critic networks
    if self._restrict_input_size > 0:
      trajectories = trajectories._replace(
          observation=trajectories.observation[:, :,
                                               self._restrict_input_size:])

    for _ in range(num_steps):
      self._session.run([self.agent_train_op, self.summary_ops],
                        feed_dict=self._get_dict(
                            trajectories, batch_size=batch_size))

    if recompute_reward:
      return new_reward, info
    else:
      return None, None

  @property
  def skill_dynamics(self):
    return self._skill_dynamics

  @property
  def observation_processor(self):
    return self._observation_processor

  def _train(self, experience, weights):
      """Returns a train op to update the agent's networks.
      This method trains with the provided batched experience.
      Args:
        experience: A time-stacked trajectory object.
        weights: Optional scalar or elementwise (per-batch-entry) importance
          weights.
      Returns:
        A train_op.
      Raises:
        ValueError: If optimizers are None and no default value was provided to
          the constructor.
      """
      squeeze_time_dim = not self._critic_network_1.state_spec
      time_steps, policy_steps, next_time_steps = (
          trajectory.experience_to_transitions(experience, squeeze_time_dim))
      actions = policy_steps.action

      trainable_critic_variables = list(object_identity.ObjectIdentitySet(
          self._critic_network_1.trainable_variables +
          self._critic_network_2.trainable_variables))

      with tf.GradientTape(watch_accessed_variables=False) as tape:
          assert trainable_critic_variables, ('No trainable critic variables to '
                                              'optimize.')
          tape.watch(trainable_critic_variables)
          self.observation_processor.build_graph(timesteps=time_steps, is_training=True)
          critic_loss = self._critic_loss_weight * self.critic_loss(
              self.observation_processor.mean,
              actions,
              next_time_steps,
              td_errors_loss_fn=self._td_errors_loss_fn,
              gamma=self._gamma,
              reward_scale_factor=self._reward_scale_factor,
              weights=weights,
              training=True)

      tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
      critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
      self._apply_gradients(critic_grads, trainable_critic_variables,
                            self._critic_optimizer)

      trainable_actor_variables = self._actor_network.trainable_variables
      trainable_obs_proc_variables = self.observation_processor.trainable_variables
      with tf.GradientTape(watch_accessed_variables=False) as tape:
          assert trainable_actor_variables, ('No trainable actor variables to '
                                             'optimize.')
          tape.watch(trainable_actor_variables + trainable_obs_proc_variables)
          actor_loss = self._actor_loss_weight * self.actor_loss(time_steps, weights=weights)

          # train the observation processor to hide information from the agent
          self.observation_processor.build_graph(timesteps=time_steps, is_training=True)
          denominator = tf.math.reduce_logsumexp(
              self.observation_processor.log_probability)
          denominator -= tf.math.log(tf.cast(tf.shape(
              self.observation_processor.mean)[0], tf.float32))
          actor_loss += self._information_hiding_weight * tf.reduce_mean(
              self.observation_processor.log_probability - denominator)

      tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
      actor_grads = tape.gradient(actor_loss, trainable_actor_variables + trainable_obs_proc_variables)
      self._apply_gradients(actor_grads, trainable_actor_variables + trainable_obs_proc_variables,
                            self._actor_optimizer)

      alpha_variable = [self._log_alpha]
      with tf.GradientTape(watch_accessed_variables=False) as tape:
          assert alpha_variable, 'No alpha variable to optimize.'
          tape.watch(alpha_variable)
          self.observation_processor.build_graph(timesteps=time_steps, is_training=True)
          alpha_loss = self._alpha_loss_weight * self.alpha_loss(
              self.observation_processor.mean, weights=weights)
      tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
      alpha_grads = tape.gradient(alpha_loss, alpha_variable)
      self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

      with tf.name_scope('Losses'):
          tf.compat.v2.summary.scalar(
              name='critic_loss', data=critic_loss, step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='actor_loss', data=actor_loss, step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='alpha_loss', data=alpha_loss, step=self.train_step_counter)

      self.train_step_counter.assign_add(1)
      self._update_target()

      total_loss = critic_loss + actor_loss + alpha_loss

      extra = SacLossInfo(
          critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss)

      return tf_agent.LossInfo(loss=total_loss, extra=extra)
