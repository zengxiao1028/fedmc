import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from src.prune_utils import PrunableLayer
class TransposableEmbedding(tf.keras.layers.Layer):
  """A Keras layer implements a transposed projection for output."""

  def __init__(self, embedding_layer: tf.keras.layers.Embedding):
    super().__init__()
    self.embeddings = embedding_layer.embeddings

  # Placing `tf.matmul` under the `call` method is important for backpropagating
  # the gradients of `self.embeddings` in graph mode.
  def call(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_recurrent_model(vocab_size=10000,
                           num_oov_buckets=1,
                           embedding_size=96,
                           latent_size=670,
                           num_layers=1,
                           name='rnn',
                           shared_embedding=False,
                           prune=True,
                           prune_sched_fn=None):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
      vocab_size: Size of vocabulary to use.
      num_oov_buckets: Number of out of vocabulary buckets.
      embedding_size: The size of the embedding.
      latent_size: The size of the recurrent state.
      num_layers: The number of layers.
      name: (Optional) string to name the returned `tf.keras.Model`.
      shared_embedding: (Optional) Whether to tie the input and output
        embeddings.

  Returns:
    `tf.keras.Model`.
  """

  if not prune_sched_fn:
    prune_sched_fn = lambda: pruning_sched.ConstantSparsity(target_sparsity=0.5, begin_step=0)

  prune_wrapper = lambda x: PrunableLayer(x, prune_sched_fn(), enable_prune=prune)

  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  inputs = tf.keras.layers.Input(shape=(None,))
  input_embedding = prune_wrapper(tf.keras.layers.Embedding(
      input_dim=extended_vocab_size, output_dim=embedding_size, mask_zero=True))
  embedded = input_embedding(inputs)
  projected = embedded

  for _ in range(num_layers):
    layer = prune_wrapper(tf.keras.layers.LSTM(latent_size, return_sequences=True))
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to input_embedding_size
    projected = prune_wrapper(tf.keras.layers.Dense(embedding_size))(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    logits = prune_wrapper(tf.keras.layers.Dense(
        extended_vocab_size, activation=None))(
            projected)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)