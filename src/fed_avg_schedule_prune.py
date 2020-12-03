# Copyright 2019, Google LLC.
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
"""An implementation of the FedAvg algorithm with learning rate schedules.

This is intended to be a somewhat minimal implementation of Federated
Averaging that allows for client and server learning rate scheduling.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Callable, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from src import prune_utils
import tensorflow_model_optimization as tfmot


# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
    """Ensures variables holding the state of `optimizer` are created."""
    delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
    model_weights = _get_weights(model)
    grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                           model_weights.trainable)
    optimizer.apply_gradients(grads_and_vars, name='server_update')
    assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> prune_utils.PruneModelWeights:
    """
    return a contrainer that contains trainable, non_trainable, and masks
    """
    return prune_utils.PruneModelWeights.from_model(model)


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
    """Structure for state on the server.
    Fields:
    -   `model`: A dictionary of the model's trainable and non-trainable
          weights.
    -   `optimizer_state`: The server optimizer variables.
    -   `round_num`: The current training round, as a float.
    """
    model = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()
    server_masks = attr.ib()
    # This is a float to avoid type incompatibility when calculating learning rate
    # schedules.



@tf.function
def server_update(model, server_optimizer, server_state, weights_delta, agr_prune_masks):
    """Updates `server_state` based on `weights_delta`, increase the round number.

    Args:
      model: A `tff.learning.Model`. (brand new)
      server_optimizer: A `tf.keras.optimizers.Optimizer`. (brand new)
      server_state: A `ServerState`, the state to be updated.
      weights_delta: An update to the trainable variables of the model.
      agr_prune_masks: aggregated prune mask

    Returns:
      An updated `ServerState`.
    """
    model_weights = _get_weights(model)
    tff.utils.assign(model_weights, server_state.model)
    # Server optimizer variables must be initialized prior to invoking this
    tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)
    #restore server masks from previous state
    #tff.utils.assign(server_masks, server_state.server_masks)

    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    if has_non_finite_weight > 0:
        return server_state

    # Apply the update to the model. We must multiply weights_delta by -1.0 to
    # view it as a gradient that should be applied to the server_optimizer.
    grads_and_vars = [
        (-1.0 * x, v) for x, v in zip(weights_delta, model_weights.trainable)
    ]

    server_optimizer.apply_gradients(grads_and_vars)

    ### use server model to compute mask (ServerMC)
    ### new_masks = update_masks(model)
    ### tff.utils.assign(model_weights.pruning_vars[1], new_masks)


    tff.utils.assign(model_weights.pruning_vars[1], agr_prune_masks)



    # update pruning step
    pruning_steps = prune_utils.get_vars_by_name(model_weights.non_trainable, 'pruning_step')
    prune_utils.assign_add(pruning_steps)

    # Create a new state based on the updated model.
    return tff.utils.update_state(
        server_state,
        model=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1.0,
        server_masks=server_state.server_masks)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.

    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
    -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
    -   `model_output`: A structure matching
        `tff.learning.Model.report_local_outputs`, reflecting the results of
        training on the input dataset.
    -   `optimizer_output`: Additional metrics or other outputs defined by the
        optimizer.
    """
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()
    optimizer_output = attr.ib()


#TODO: update mask when needed
def update_masks(model, keep_sign):
    """
    :param model: a tff.learning.model or a enhanced one
    """

    def update_mask(weights, mask, sparsity):
        abs_weights = tf.math.abs(weights)

        k = tf.dtypes.cast(
            tf.math.round(
                tf.dtypes.cast(tf.size(abs_weights), tf.float32) *
                (1 - sparsity)), tf.int32)

        # orig_shape = tf.shape(abs_weights)
        # z = -tf.math.log(-tf.math.log(tf.random.uniform(orig_shape, 0, 1)))
        # sel = tf.reshape(abs_weights + z, [-1])
        # _, indices = tf.math.top_k(sel, k, sorted=False)
        # indices = tf.reshape(indices, [-1, 1])
        # new_mask = tf.dtypes.cast(tf.scatter_nd(indices, tf.ones(tf.shape(indices)[0]), (tf.shape(sel)[0],) ), weights.dtype)
        # new_mask = tf.reshape(new_mask, orig_shape)

        # Sort the entire array
        values, _ = tf.math.top_k(
            tf.reshape(abs_weights, [-1]), k=tf.size(abs_weights))
        # Grab the (k-1)th value
        current_threshold = tf.gather(values, k - 1)
        new_mask = tf.dtypes.cast(
            tf.math.greater_equal(abs_weights, current_threshold), weights.dtype)

        if keep_sign:
            return tf.multiply(tf.math.sign(weights), new_mask)
        else:
            return new_mask

    keras_model = model._model._keras_model
    weight_vars, mask_vars, target_sparsities = [], [], []
    for layer in keras_model.layers:
        if hasattr(layer, 'pruning_vars'):
            for w_var, m_var, sparsity in layer.pruning_vars:
                weight_vars.append(w_var)
                mask_vars.append(m_var)
                _, sparsity = layer.pruning_schedule(layer.pruning_step)
                target_sparsities.append(sparsity)

    new_masks =tf.nest.map_structure(lambda a, b, c: update_mask(a, b, c),
                            weight_vars, mask_vars, target_sparsities)

    return new_masks



def create_client_update_fn():
    """Returns a tf.function for the client_update.

    This "create" fn is necesessary to prevent
    "ValueError: Creating variables on a non-first call to a function decorated
    with tf.function" errors due to the client optimizer creating variables. This
    is really only needed because we test the client_update function directly.
    """

    @tf.function
    def client_update(model,
                      dataset,
                      initial_weights,
                      client_optimizer,
                      client_weight_fn=None):
        """Updates client model.

        Args:
          model: A `tff.learning.Model`.
          dataset: A 'tf.data.Dataset'.
          initial_weights: A `tff.learning.ModelWeights` from server.
          client_optimizer: A `tf.keras.optimizer.Optimizer` object.
          client_weight_fn: Optional function that takes the output of
            `model.report_local_outputs` and returns a tensor that provides the
            weight in the federated average of model deltas. If not provided, the
            default is the total number of examples processed on device.

        Returns:
          A 'ClientOutput`.
        """

        model_weights = _get_weights(model)
        tff.utils.assign(model_weights, initial_weights)

        num_examples = tf.constant(0, dtype=tf.int32)
        for batch in iter(dataset):
            with tf.GradientTape() as tape:
                output = model.forward_pass(batch)
            grads = tape.gradient(output.loss, model_weights.trainable)
            grads_and_vars = zip(grads, model_weights.trainable)
            client_optimizer.apply_gradients(grads_and_vars)
            num_examples += tf.shape(output.predictions)[0]

        new_masks = update_masks(model,keep_sign=False)

        aggregated_outputs = model.report_local_outputs()
        weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                              model_weights.trainable,
                                              initial_weights.trainable)
        weights_delta, has_non_finite_weight = (
            tensor_utils.zero_all_if_any_non_finite(weights_delta))

        if has_non_finite_weight > 0:
            client_weight = tf.constant(0, dtype=tf.float32)
        elif client_weight_fn is None:
            client_weight = tf.cast(num_examples, dtype=tf.float32)
        else:
            client_weight = client_weight_fn(aggregated_outputs)

        return ClientOutput(
            weights_delta, client_weight, aggregated_outputs,
            collections.OrderedDict([('num_examples', num_examples),
                                    ('new_masks', new_masks)]) )

    return client_update


def build_server_init_fn(
        model_fn: ModelBuilder,
        server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]):
    """Builds a `tff.tf_computation` that returns the initial `ServerState`.

    The attributes `ServerState.model` and `ServerState.optimizer_state` are
    initialized via their constructor functions. The attribute
    `ServerState.round_num` is set to 0.0.

    Args:
      model_fn: A no-arg function that returns a `tff.learning.Model`.
      server_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer`.

    Returns:
      A `tff.tf_computation` that returns initial `ServerState`.
    """

    @tff.tf_computation
    def server_init_tf():
        server_optimizer = server_optimizer_fn()
        model = model_fn()
        _initialize_optimizer_vars(model, server_optimizer)
        model_weights = _get_weights(model)

        #create server masks for aggregating client masks
        server_masks = tf.nest.map_structure(lambda a: tf.zeros_like(a, dtype=a.dtype), model_weights.pruning_vars[1])

        #server_masks = model_weights.pruning_vars[1]
        return ServerState(
            model=model_weights,
            optimizer_state=server_optimizer.variables(),
            round_num=0.0,
            server_masks=server_masks)

    return server_init_tf


def build_fed_avg_process(
        model_fn: ModelBuilder,
        client_optimizer_fn: OptimizerBuilder,
        client_lr: Union[float, LRScheduleFn] = 0.1,
        server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
        server_lr: Union[float, LRScheduleFn] = 1.0,
        client_weight_fn: Optional[ClientWeightFn] = None,
) -> tff.templates.IterativeProcess:
    """Builds the TFF computations for optimization using federated averaging.

    Args:
      model_fn: A no-arg function that returns a `tff.learning.Model`.
      client_optimizer_fn: A function that accepts a `learning_rate` keyword
        argument and returns a `tf.keras.optimizers.Optimizer` instance.
      client_lr: A scalar learning rate or a function that accepts a float
        `round_num` argument and returns a learning rate.
      server_optimizer_fn: A function that accepts a `learning_rate` argument and
        returns a `tf.keras.optimizers.Optimizer` instance.
      server_lr: A scalar learning rate or a function that accepts a float
        `round_num` argument and returns a learning rate.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the weight
        in the federated average of model deltas. If not provided, the default is
        the total number of examples processed on device.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    client_lr_schedule = client_lr
    if not callable(client_lr_schedule):
        client_lr_schedule = lambda round_num: client_lr

    server_lr_schedule = server_lr
    if not callable(server_lr_schedule):
        server_lr_schedule = lambda round_num: server_lr

    dummy_model = model_fn()

    server_init_tf = build_server_init_fn(
        model_fn,
        # Initialize with the learning rate for round zero.
        lambda: server_optimizer_fn(server_lr_schedule(0)))
    server_state_type = server_init_tf.type_signature.result
    model_weights_type = server_state_type.model
    round_num_type = server_state_type.round_num
    tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
    model_input_type = tff.SequenceType(dummy_model.input_spec)

    @tff.tf_computation(model_input_type, model_weights_type, round_num_type)
    def client_update_fn(tf_dataset, initial_model_weights, round_num):
        client_lr = client_lr_schedule(round_num)
        client_optimizer = client_optimizer_fn(client_lr)
        client_update = create_client_update_fn()
        return client_update(model_fn(), tf_dataset, initial_model_weights,
                             client_optimizer, client_weight_fn)

    @tff.tf_computation(server_state_type, model_weights_type.trainable, server_state_type.server_masks)
    def server_update_fn(server_state, model_delta, agr_prune_masks):
        model = model_fn()
        server_lr = server_lr_schedule(server_state.round_num)
        server_optimizer = server_optimizer_fn(server_lr)
        # We initialize the server optimizer variables to avoid creating them
        # within the scope of the tf.function server_update.
        _initialize_optimizer_vars(model, server_optimizer)

        return server_update(model, server_optimizer, server_state, model_delta, agr_prune_masks)

    @tff.federated_computation(
        tff.type_at_server(server_state_type),
        tff.type_at_clients(tf_dataset_type))
    def run_one_round(server_state, federated_dataset):
        """Orchestration logic for one round of computation.

        Args:
          server_state: A `ServerState`.
          federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

        Returns:
          A tuple of updated `ServerState` and the result of
          `tff.learning.Model.federated_output_computation`.
        """
        client_model = tff.federated_broadcast(server_state.model)
        client_round_num = tff.federated_broadcast(server_state.round_num)

        client_outputs = tff.federated_map(
            client_update_fn,
            (federated_dataset, client_model, client_round_num))

        client_weight = client_outputs.client_weight

        # avarage_trainable_weights
        model_delta = tff.federated_mean(
            client_outputs.weights_delta,
            weight=client_weight)

        # average mask
        aggregated_model_masks = tff.federated_mean(
            client_outputs.optimizer_output['new_masks'],
            weight=client_weight)

        # apply model delta and average mask to server
        server_state = tff.federated_map(server_update_fn,
                                         (server_state, model_delta, aggregated_model_masks))

        aggregated_outputs = dummy_model.federated_output_computation(client_outputs.model_output)
        if aggregated_outputs.type_signature.is_struct():
            aggregated_outputs = tff.federated_zip(aggregated_outputs)

        return server_state, aggregated_outputs, aggregated_model_masks

    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value(server_init_tf(), tff.SERVER)

    return tff.templates.IterativeProcess(
        initialize_fn=initialize_fn, next_fn=run_one_round)
