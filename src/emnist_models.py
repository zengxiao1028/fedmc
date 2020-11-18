import tensorflow as tf
from src import prune_utils
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched

def create_model(only_digits=True, prune=True):
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
    if prune:
        layers = [prune_utils.PrunableLayer(l, pruning_sched.ConstantSparsity(0.5, 50)) for l in layers]
    model = tf.keras.models.Sequential(layers)
    return model


if __name__ == '__main__':
    model = create_model(only_digits=False)
    for v in model.model_masks:
        print(v.name)
    model.summary()
