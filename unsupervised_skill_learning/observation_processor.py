"""Observation Processing Network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ObservationProcessor:

  def __init__(
      self,
      observation_size,
      restrict_observation=0,
      normalize_observations=False,
      # network properties
      fc_layer_params=(256, 256),
      network_type='default',
      num_components=1,
      fix_variance=False,
      reweigh_batches=False,
      graph=None,
      scope_name='observation_processor'):

    self._observation_size = observation_size
    self._normalize_observations = normalize_observations
    self._restrict_observation = restrict_observation
    self._reweigh_batches = reweigh_batches

    # tensorflow requirements
    if graph is not None:
      self._graph = graph
    else:
      self._graph = tf.compat.v1.get_default_graph()
    self._scope_name = scope_name

    # dynamics network properties
    self._fc_layer_params = fc_layer_params
    self._network_type = network_type
    self._num_components = num_components
    self._fix_variance = fix_variance
    if not self._fix_variance:
      self._std_lower_clip = 0.3
      self._std_upper_clip = 10.0

    self._use_placeholders = False
    self.log_probability = None
    self.mean = None
    self._session = None
    self._use_modal_mean = False

    # saving/restoring variables
    self._saver = None

  def _get_distribution(self, out):
    if self._num_components > 1:
      self.logits = tf.compat.v1.layers.dense(
          out, self._num_components, name='logits', reuse=tf.compat.v1.AUTO_REUSE)
      means, scale_diags = [], []
      for component_id in range(self._num_components):
        means.append(
            tf.compat.v1.layers.dense(
                out,
                self._observation_size,
                name='mean_' + str(component_id),
                reuse=tf.compat.v1.AUTO_REUSE))
        if not self._fix_variance:
          scale_diags.append(
              tf.clip_by_value(
                  tf.compat.v1.layers.dense(
                      out,
                      self._observation_size,
                      activation=tf.nn.softplus,
                      name='stddev_' + str(component_id),
                      reuse=tf.compat.v1.AUTO_REUSE), self._std_lower_clip,
                  self._std_upper_clip))
        else:
          scale_diags.append(
              tf.fill([tf.shape(out)[0], self._observation_size], 1.0))

      self.means = tf.stack(means, axis=1)
      self.scale_diags = tf.stack(scale_diags, axis=1)
      return tfp.distributions.MixtureSameFamily(
          mixture_distribution=tfp.distributions.Categorical(
              logits=self.logits),
          components_distribution=tfp.distributions.MultivariateNormalDiag(
              loc=self.means, scale_diag=self.scale_diags))

    else:
      mean = tf.compat.v1.layers.dense(
          out, self._observation_size, name='mean', reuse=tf.compat.v1.AUTO_REUSE)
      if not self._fix_variance:
        stddev = tf.clip_by_value(
            tf.compat.v1.layers.dense(
                out,
                self._observation_size,
                activation=tf.nn.softplus,
                name='stddev',
                reuse=tf.compat.v1.AUTO_REUSE), self._std_lower_clip,
            self._std_upper_clip)
      else:
        stddev = tf.fill([tf.shape(out)[0], self._observation_size], 1.0)
      return tfp.distributions.MultivariateNormalDiag(
          loc=mean, scale_diag=stddev)

  def _default_graph(self, timesteps):
    out = timesteps
    for idx, layer_size in enumerate(self._fc_layer_params):
      out = tf.compat.v1.layers.dense(
          out,
          layer_size,
          activation=tf.nn.relu,
          name='hid_' + str(idx),
          reuse=tf.compat.v1.AUTO_REUSE)

    return self._get_distribution(out)

  def _get_dict(self,
                input_data,
                batch_size=-1,
                batch_weights=None,
                batch_norm=False):
    if batch_size > 0:
      shuffled_batch = np.random.permutation(len(input_data))[:batch_size]
    else:
      shuffled_batch = np.arange(len(input_data))

    # if we are noising the input, it is better to create a new copy of the numpy arrays
    batched_input = input_data[shuffled_batch, :]

    if self._reweigh_batches and batch_weights is not None:
      example_weights = batch_weights[shuffled_batch]

    return_dict = {
        self.timesteps_pl: batched_input,
    }
    if self._normalize_observations:
      return_dict[self.is_training_pl] = batch_norm
    if self._reweigh_batches and batch_weights is not None:
      return_dict[self.batch_weights] = example_weights

    return return_dict

  def _get_run_dict(self, input_data):
    return_dict = {
        self.timesteps_pl: input_data,
    }
    if self._normalize_observations:
      return_dict[self.is_training_pl] = False

    return return_dict

  def make_placeholders(self):
    self._use_placeholders = True
    with self._graph.as_default(), tf.compat.v1.variable_scope(self._scope_name):
      self.timesteps_pl = tf.compat.v1.placeholder(
          tf.float32, shape=(None, self._observation_size), name='timesteps_pl')
      if self._normalize_observations:
        self.is_training_pl = tf.compat.v1.placeholder(tf.bool, name='batch_norm_pl')
      if self._reweigh_batches:
        self.batch_weights = tf.compat.v1.placeholder(
            tf.float32, shape=(None,), name='importance_sampled_weights')

  def set_session(self, session=None, initialize_or_restore_variables=False):
    if session is None:
      self._session = tf.Session(graph=self._graph)
    else:
      self._session = session

    # only initialize uninitialized variables
    if initialize_or_restore_variables:
      if tf.io.gfile.exists(self._save_prefix):
        self.restore_variables()
      with self._graph.as_default():
        var_list = tf.compat.v1.global_variables(
        ) + tf.compat.v1.local_variables()
        is_initialized = self._session.run(
            [tf.compat.v1.is_variable_initialized(v) for v in var_list])
        uninitialized_vars = []
        for flag, v in zip(is_initialized, var_list):
          if not flag:
            uninitialized_vars.append(v)

        if uninitialized_vars:
          self._session.run(
              tf.compat.v1.variables_initializer(uninitialized_vars))

  def build_graph(self,
                  timesteps=None,
                  is_training=None):
    with self._graph.as_default(), tf.compat.v1.variable_scope(
        self._scope_name, reuse=tf.compat.v1.AUTO_REUSE):
      if self._use_placeholders:
        timesteps = self.timesteps_pl
        if self._normalize_observations:
          is_training = self.is_training_pl

      if self._restrict_observation > 0:
        timesteps = timesteps[:, self._restrict_observation:]

      if self._normalize_observations:
        timesteps = tf.compat.v1.layers.batch_normalization(
            timesteps,
            training=is_training,
            name='input_normalization',
            reuse=tf.compat.v1.AUTO_REUSE)

      self.base_distribution = self._default_graph(timesteps)

      # if building multiple times, be careful about which log_prob you are optimizing
      samples = self.base_distribution.sample()
      self.log_probability = self.base_distribution.log_prob(samples)
      self.mean = self.base_distribution.mean()

      return self.log_probability

  def create_saver(self, save_prefix):
    if self._saver is not None:
      return self._saver
    else:
      with self._graph.as_default():
        self._variable_list = {}
        for var in tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self._scope_name):
          self._variable_list[var.name] = var
        self._saver = tf.compat.v1.train.Saver(
            self._variable_list, save_relative_paths=True)
        self._save_prefix = save_prefix

  def save_variables(self, global_step):
    if not tf.io.gfile.exists(self._save_prefix):
      tf.io.gfile.makedirs(self._save_prefix)

    self._saver.save(
        self._session,
        os.path.join(self._save_prefix, 'ckpt'),
        global_step=global_step)

  def restore_variables(self):
    self._saver.restore(self._session,
                        tf.compat.v1.train.latest_checkpoint(self._save_prefix))

  # all functions here-on require placeholders----------------------------------
  def get_log_prob(self, timesteps):
    if not self._use_placeholders:
      return

    return self._session.run(
        self.log_probability,
        feed_dict=self._get_dict(
            timesteps, batch_norm=False))

  def predict_state(self, timesteps):
    if not self._use_placeholders:
      return

    if self._use_modal_mean:
      all_means, modal_mean_indices = self._session.run(
          [self.means, tf.argmax(self.logits, axis=1)],
          feed_dict=self._get_run_dict(timesteps))
      pred_state = all_means[[
          np.arange(all_means.shape[0]), modal_mean_indices
      ]]
    else:
      pred_state = self._session.run(
          self.mean, feed_dict=self._get_run_dict(timesteps))

    return pred_state

  @property
  def trainable_variables(self):
    return tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope_name)
