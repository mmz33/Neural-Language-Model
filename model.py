import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq

class Model(object):
  """
  RNN language model
  """

  def __init__(self, args, is_training=True):
    self.batch_size = args.batch_size
    self.num_steps = args.num_steps # time steps
    self.model = args.model # e.g rnn, lstm, gru
    self.optimizer = args.optimizer
    self.dropout = args.dropout

    with_gpu = args.with_gpu and tf.test.is_gpu_available() and tf.test.is_built_with_cuda()

    assert 0 <= self.dropout <= 1, 'dropout should be in the range [0,1]'

    self.num_units = args.num_units # number of hidden units
    rnn_cell = tf.nn.rnn_cell # tf build-in cells
    self.vocab_size = args.vocab_size

    assert self.vocab_size is not None, 'vocabulary is not created or read'

    if not is_training:
      self.batch_size = 1
      self.num_steps = 1

    # cell_fn: tensorflow cell function
    if self.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif self.model == 'lstm':
      cell_fn = rnn_cell.LSTMCell
    elif self.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    else:
      raise Exception('{} is not supported'.format(self.model))

    # placeholders for data
    self.input_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps]) # (B,T)
    self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_steps]) # (B,T)

    # create rnn layers cells

    # if with_gpu:
    #   with tf.device('/gpu:0'):
    rnn_layers = []
    for l in range(args.num_layers):
      if self.model == 'lstm':
        cell = cell_fn(args.num_units, forget_bias=0.0)
      else:
        cell = cell_fn(args.num_units)
      rnn_layers.append(cell)
    lm_cell = rnn_cell.MultiRNNCell(rnn_layers) # construct layer cells sequentially

    # initial hidden state
    self.initial_lm_state = lm_cell.zero_state(self.batch_size, tf.float32)

    # input embedding
    with tf.device('/cpu:0'):
      # assume for simplicity that the input feature dimension is equal to the number of hidden units
      embedding = tf.get_variable('embedding', shape=[self.vocab_size, self.num_units])
      inputs = tf.nn.embedding_lookup(embedding, self.input_data) # word feature vector (B,T,D)

    # apply dropout
    if is_training and self.dropout > 0:
      inputs = tf.nn.dropout(inputs, 1 - self.dropout)

    # split input into list

    # at each time step, we feed (B,1,D) as input
    inputs = tf.split(inputs, self.num_steps, 1) # list of (B,1 D) of size T

    # convert each element to (B, D) so inputs become a list of (B,D) and size T
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    # the value of state is updated after processing each batch of words
    output, state = tf.nn.static_rnn(lm_cell, inputs, self.initial_lm_state)

    # output is a list of size T and elements shape (B,D)
    # concat on axis 1 to become of shape (B,T*D)
    lm_outputs = tf.concat(output, 1)

    # result shape (B*T,D)
    lm_outputs = tf.reshape(lm_outputs, [-1, self.num_units])

    # output_layer params
    softmax_w = tf.get_variable("softmax_w", [self.num_units, self.vocab_size])
    softmax_b = tf.get_variable("softmax_b", [self.vocab_size])

    # the LSTM output can be used to make next word predictions
    # result shape (B*T, V)
    logits = tf.matmul(lm_outputs, softmax_w) + softmax_b

    # compute log perplexity
    self.loss = legacy_seq2seq.sequence_loss_by_example(
      logits=[logits],
      targets=[tf.reshape(self.targets, [-1])],
      weights=[tf.ones([self.batch_size * self.num_steps])]
    )

    # cost is the log PP
    self.cost = tf.reduce_sum(self.loss) / self.batch_size
    self.final_state = state

    if not is_training:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    vars_ = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, vars_),
                                      args.grad_clip)

    if self.optimizer == 'momentum':
      self.optimizer = tf.train.MomentumOptimizer(self.lr, 0.95)
    else:
      self.optimizer = tf.train.GradientDescentOptimizer(self.lr)

    self.train_op = self.optimizer.apply_gradients(zip(grads, vars_))

  def assign_lr(self, sess, lr_value):
    """Assign lr_value to the learning rate variable
      e.g this can be used in case we are using lr decay during training

    :param sess: tf.Session
    :param lr_value: A float, the new value for the learning rate
    """

    sess.run(tf.assign(self.lr, lr_value))
