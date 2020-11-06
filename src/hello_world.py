import nest_asyncio
nest_asyncio.apply()

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

@tff.federated_computation
def hello_world():
  return 'Hello, World!'

print(hello_world())

@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)

print( get_average_temperature([68.5, 70.3, 69.8]) )