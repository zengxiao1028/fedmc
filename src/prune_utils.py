import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.common_libs import structure
import tensorflow as tf
import attr
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_impl

def get_vars_by_name(variables, name):
    vars = []
    for v in variables:
        if name in v.name:
            vars.append(v)
    return vars

@attr.s(eq=False, frozen=True, slots=True)
class PruneModelWeights(tff.learning.ModelWeights):
    """
    A class for storing pruning-related variables in additional to model vars.
    """
    prune_masks = attr.ib()

    @classmethod
    def from_model(cls, model):
        py_typecheck.check_type(model, (model_lib.Model, tf.keras.Model))
        prune_masks = get_vars_by_name(model.non_trainable_variables, 'prune_mask')
        return cls(model.trainable_variables, model.non_trainable_variables, prune_masks)

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



def assign_add(tensors):
    if isinstance(tensors, structure.Struct):
        return tf.group(*structure.flatten(
            structure.map_structure(lambda a: a.assign(tf.math.add(a,1)), tensors)))
    else:
        return tf.group(*tf.nest.flatten(
            tf.nest.map_structure(lambda a: a.assign(tf.math.add(a,1)), tensors)))