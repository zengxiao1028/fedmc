import tensorflow as tf
from src import prune_utils
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from absl import logging

def create_model(only_digits=True, prune=True, prune_schedule_fn=None):
    """
    Create a CNN model for emnist
    """
    data_format = 'channels_last'
    layers = [tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                     data_format=data_format, input_shape=(28, 28, 1)),
              tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                     data_format=data_format),
              tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
              tf.keras.layers.Dropout(0.25),
              tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dropout(0.5),
              tf.keras.layers.Dense(10 if only_digits else 62, activation=tf.nn.softmax)
              ]
    if not prune_schedule_fn:
        #prune_schedule_fn = lambda : pruning_sched.ConstantSparsity(0.5, 0)
        prune_schedule_fn = lambda: pruning_sched.PolynomialDecay(0, 0.5, begin_step=5,
                                                                  end_step=300, frequency=1,
                                                                  power=2)
    tmp_layers = []
    for l in layers:
        if isinstance(l, tf.keras.layers.Dropout):
            if prune:
                #logging.info('skip dropout layer when enable pruning')
                pass
            else:
                tmp_layers.append(l)
        else:
            tmp_layers.append(prune_utils.PrunableLayer(l, prune_schedule_fn(), enable_prune=prune))
    layers = tmp_layers

    model = tf.keras.models.Sequential(layers)
    return model


if __name__ == '__main__':
    model = create_model(only_digits=False)
    for v in model.model_masks:
        print(v.name)
    model.summary()
