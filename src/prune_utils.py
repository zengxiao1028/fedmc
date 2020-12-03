import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.common_libs import structure
import tensorflow as tf
import attr
keras = tf.keras
K = keras.backend
from tensorflow_model_optimization.python.core.keras import utils
import numpy as np
import inspect
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched

def get_vars_by_name(variables, name):
    vars = []
    for v in variables:
        if name in v.name:
            vars.append(v)
    return vars

def get_pruning_vars(model):
    """
    Args:
        model: tff model
    """
    keras_model = model._model._keras_model
    weight_vars, mask_vars = [], []
    for layer in keras_model.layers:
        if hasattr(layer, 'pruning_vars'):
            for w_var, m_var, threshold in layer.pruning_vars:
                weight_vars.append(w_var)
                mask_vars.append(m_var)

    return weight_vars, mask_vars


@attr.s(eq=False, frozen=True, slots=True)
class PruneModelWeights(tff.learning.ModelWeights):
    """
    A class for storing pruning-related variables in additional to model vars.
    """
    pruning_vars = attr.ib()

    @classmethod
    def from_model(cls, model):
        py_typecheck.check_type(model, (model_lib.Model, tf.keras.Model))
        pruning_vars = get_pruning_vars(model)
        return cls(model.trainable_variables, model.non_trainable_variables, pruning_vars)

    #deprecated
    @classmethod
    def from_tff_result(cls, struct):
        py_typecheck.check_type(struct, structure.Struct)
        trainables = [value for _, value in structure.iter_elements(struct.trainable)]
        non_trainables = [value for _, value in structure.iter_elements(struct.non_trainable)]
        prune_masks = get_vars_by_name(non_trainables, 'prune_mask')
        return cls(trainables, non_trainables, prune_masks)


class PrunableLayer(pruning_wrapper.PruneLowMagnitude):
    """
    Augment a keras layer into a prunable layer
    """

    def __init__(self,
                 layer,
                 pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
                 block_size=(1, 1),
                 block_pooling_type='AVG',
                 enable_prune=True,
                 **kwargs):
        super(PrunableLayer, self).__init__(layer, pruning_schedule, block_size, block_pooling_type, **kwargs)
        self.enable_prune = enable_prune


    def build(self, input_shape):
        super(pruning_wrapper.PruneLowMagnitude, self).build(input_shape)

        weight_vars, mask_vars, threshold_vars = [], [], []

        self.prunable_weights = self.layer.get_prunable_weights()

        # For each of the prunable weights, add mask and threshold variables
        for weight in self.prunable_weights:
            mask = self.add_variable(
                'prune_mask',
                shape=weight.shape,
                initializer=tf.keras.initializers.get('ones'),
                dtype=weight.dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)
            threshold = self.add_variable(
                'threshold',
                shape=[],
                initializer=tf.keras.initializers.get('zeros'),
                dtype=weight.dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            weight_vars.append(weight)
            mask_vars.append(mask)
            threshold_vars.append(threshold)
        self.pruning_vars = list(zip(weight_vars, mask_vars, threshold_vars))

        # Add a scalar tracking the number of updates to the wrapped layer.
        self.pruning_step = self.add_variable(
            'pruning_step',
            shape=[],
            initializer=tf.keras.initializers.Constant(0), #start from 0 to avoid error
            dtype=tf.int64,
            trainable=False)

        def training_step_fn():
            return self.pruning_step

        # Create a pruning object
        self.pruning_obj = pruning_impl.Pruning(
            training_step_fn=training_step_fn,
            pruning_vars=self.pruning_vars,
            pruning_schedule=self.pruning_schedule,
            block_size=self.block_size,
            block_pooling_type=self.block_pooling_type)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        if self.enable_prune:
            # def add_update():
            #     with tf.control_dependencies([
            #         tf.debugging.assert_greater_equal(
            #             self.pruning_step,
            #             np.int64(0),
            #             message=self._PRUNE_CALLBACK_ERROR_MSG)
            #     ]):
            #         # with tf.control_dependencies(
            #         #         [self.pruning_obj.conditional_mask_update()]):
            #             return tf.no_op('update')
            #
            # def no_op():
            #     return tf.no_op('no_update')
            #
            # update_op = utils.smart_cond(training, add_update, no_op)
            # self.add_update(update_op)
            # Always execute the op that performs weights = weights * mask
            # Relies on UpdatePruningStep callback to ensure the weights
            # are sparse after the final backpropagation.
            #
            # self.add_update does nothing during eager execution.
            self.add_update(self.pruning_obj.weight_mask_op())
            # TODO(evcu) remove this check after dropping py2 support. In py3 getargspec
            # is deprecated.
        if hasattr(inspect, 'getfullargspec'):
            args = inspect.getfullargspec(self.layer.call).args
        else:
            args = inspect.getargspec(self.layer.call).args
        # Propagate the training bool to the underlying layer if it accepts
        # training as an arg.
        if 'training' in args:
            return self.layer.call(inputs, training=training)

        return self.layer.call(inputs)




def assign_add(tensors):
    if isinstance(tensors, structure.Struct):
        return tf.group(*structure.flatten(
            structure.map_structure(lambda a: a.assign(tf.math.add(a,1)), tensors)))
    else:
        return tf.group(*tf.nest.flatten(
            tf.nest.map_structure(lambda a: a.assign(tf.math.add(a,1)), tensors)))

def assign_addv2(tensors1, tensors2):
    if isinstance(tensors1, structure.Struct):
        return tf.group(*structure.flatten(
            structure.map_structure(lambda a, b: a.assign(tf.math.add(a, b)), tensors1, tensors2)))
    else:
        return tf.group(*tf.nest.flatten(
            tf.nest.map_structure(lambda a, b: a.assign(tf.math.add(a, b)), tensors1, tensors2)))