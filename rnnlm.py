import tensorflow as tf
import argparse
import time
from dataset_reader import DatasetReader
import numpy as np
import os.path
import _pickle as cPickle
import codecs
from model import Model

"""This file is the main entry point"""

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

def main():
  parser = argparse.ArgumentParser()
  # Flags for train mode (--task='train')
  parser.add_argument('--task', type=str, default='train',
                      help="'train' or 'test'.")
  parser.add_argument('--train_file', type=str, default=None,
                      help="training data path")
  parser.add_argument('--dev_file', type=str, default=None,
                      help="development/validation data path")
  parser.add_argument('--vocab_file', type=str, default=None,
                      help="vocabulary file path")
  parser.add_argument('--save_dir', type=str, default='models',
                      help='directory to store model checkpoints')  # Also needed for testing!
  parser.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
  parser.add_argument('--num_units', type=int, default=50,
                      help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=1,
                      help='number of layers in the RNN')
  parser.add_argument('--dropout', type=int, default=0.0,
                      help='dropout rate')

  # Other flags
  parser.add_argument('--output', '-o', type=str, default='train.log',
                      help='output file')
  parser.add_argument('--num_epochs', type=int, default=10,
                      help='number of training epochs')
  parser.add_argument('--decay_rate', type=float, default=0.5,
                      help='the decay of the learning rate')
  parser.add_argument('--model', type=str, default='lstm',
                      help='rnn, gru, or lstm')
  parser.add_argument('--batch_size', type=int, default=20,
                      help='minibatch size')
  parser.add_argument('--num_steps', type=int, default=20,
                      help='BPTT sequence length')
  parser.add_argument('--validation_interval', type=int, default=1,
                      help='validation interval')
  parser.add_argument('--init_scale', type=float, default=0.1,
                      help='initial weight scale')
  parser.add_argument('--grad_clip', type=float, default=5.0,
                      help='maximum permissible norm of the gradient')
  parser.add_argument('--optimizer', type=str, default='sgd',
                      help='sgd, momentum, or adagrad')
  parser.add_argument('--with_gpu', type=bool, default=False),

  # Flags for test mode (--task='test').
  parser.add_argument('--test_file', type=str, default='',
                      help="test file.")
  parser.add_argument('--compute_ppl', type=str, default='',
                      help='compute perplexity if the input sentence.')

  args = parser.parse_args()
  if args.task == 'train':
    train(args)
  elif args.task == 'test':
    test(args)
  else:
    print('Unknown task %s . Only "train" or "test" are supported.' % args.task)

def run_epoch(sess, model, data, dataset_reader, eval_op, verbose=False):
  """Run training loop for one epoch

  :param sess: tf.Session()
  :param model: Model, e.g lstm, rnn, etc
  :param data: dataset, e.g train data, test data
  :param dataset_reader: DatasetReader
  :param eval_op: train operation used for evaluation
  :param verbose: If True, then output training logs else don't
  """

  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  total_cost = 0.0
  num_iters = 0
  state = tf.get_default_session().run(model.initial_lm_state)
  for step, (x, y) in enumerate(dataset_reader.data_iterator(data, model.batch_size, model.num_steps)):
    cost, state, _ = sess.run([model.cost, model.final_state, eval_op],
                              {model.input_data: x,
                               model.targets: y,
                               model.initial_lm_state: state})
    total_cost += cost
    num_iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("(%.2f %%) perplexity: %.3f speed: %.0f word/sec" %
            (step * 1.0 / epoch_size, np.exp(total_cost / num_iters),
             num_iters * model.batch_size / (time.time() - start_time)))

  return np.exp(total_cost/num_iters)

def train(args):
  """Train the data train corpus

  :param args: system args
  """

  start = time.time()
  save_dir = args.save_dir
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
    cPickle.dump(args, f)

  data_reader = DatasetReader(args)
  train_data = data_reader.train_data

  assert train_data is not None, 'training data is not read!'

  print('Number of train running words: {}'.format(len(train_data)))

  dev_data = data_reader.dev_data
  if dev_data:
    print('Number of dev set running words: {}'.format(len(dev_data)))

  out_file = os.path.join(args.save_dir, args.output)
  fout = codecs.open(out_file, "w", encoding="UTF-8")

  args.vocab_size = data_reader.vocab_size
  print('vocab size: {}'.format(args.vocab_size))
  fout.write('vocab size: {}\n'.format(str(args.vocab_size)))

  print('Start training....')

  with tf.Graph().as_default(), tf.Session(config=gpu_config if args.with_gpu else None) as sess:

    if args.init_scale:
      initializer = tf.random_uniform_initializer(-args.init_scale, +args.init_scale)
    else:
      initializer = tf.glorot_uniform_initializer()

    # build models
    with tf.variable_scope('train_model', reuse=None, initializer=initializer):
      m_train = Model(args)

    if dev_data:
      # reuse the same embedding matrix
      with tf.variable_scope('train_model', reuse=True, initializer=initializer):
        m_dev = Model(args, is_training=False)
    else:
      m_dev = None

    # save only the last model
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    tf.global_variables_initializer().run()

    best_pp = 10000000.0 # only used when we have dev

    e = 0
    decay_counter = 1
    lr = args.lr
    while e < args.num_epochs:
      # apply lr decay after epoch 4
      if e > 4:
        lr_decay = args.decay_rate ** decay_counter
        lr *= lr_decay
        decay_counter += 1

      print('Epoch: %d' % (e+1))

      m_train.assign_lr(sess, lr)
      print('Learning rate: %.3f' % sess.run(m_train.lr))

      fout.write("Epoch: %d\n" % (e + 1))
      fout.write("Learning rate: %.3f\n" % sess.run(m_train.lr))

      train_pp = run_epoch(sess,
                           m_train,
                           train_data,
                           data_reader,
                           m_train.train_op,
                           verbose=True)

      print('Train Perplexity: {}'.format(train_pp))
      fout.write("Train Perplexity: %.3f\n" % train_pp)

      if m_dev:
        dev_pp = run_epoch(sess,
                           m_dev,
                           dev_data,
                           data_reader,
                           tf.no_op())

        print("Valid Perplexity: %.3f\n" % dev_pp)
        fout.write("Valid Perplexity: %.3f\n" % dev_pp)

        if dev_pp < best_pp:
          print("Achieve highest perplexity on dev set, save model.")
          checkpoint_path = os.path.join(save_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=e)
          print("model saved to {}".format(checkpoint_path))
          best_pp = dev_pp
        else:
          break

      fout.flush()

      e += 1

    print("Training time: %.0f" % (time.time() - start))
    fout.write("Training time: %.0f\n" % (time.time() - start))
    fout.flush()

def test(test_args):
  """Computes test perplexity for test data

  :param test_args: system args
  """

  start = time.time()

  # load hyperparameters and other flags
  with open(os.path.join(test_args.save_dir, 'config.pkl')) as f:
    args = cPickle.load(f)

  data_reader = DatasetReader(args, train=False)
  test_data = data_reader.test_data

  assert test_data is not None, 'test data is not read!'

  args.vocab_size = data_reader.vocab_size
  print('vocab_size: {}'.format(args.vocab_size))

  print('Start testing...')

  with tf.Graph().as_default(), tf.Session(config=gpu_config if args.with_gpu else None) as sess:

    # TODO: do we need this initializer?
    # if args.init_scale:
    #   initializer = tf.random_uniform_initializer(-args.init_scale, +args.init_scale)
    # else:
    #   initializer = tf.glorot_uniform_initializer()

    with tf.variable_scope('test_model', reuse=None):
      m_test = Model(args, is_training=False)

    saver = tf.train.Saver(tf.global_variables())
    tf.global_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state(args.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

    test_pp = run_epoch(sess,
                        m_test,
                        test_data,
                        data_reader,
                        tf.no_op())

    print('Test Perplexity: %.3f'.format(test_pp))
    print("Test time: %.0f" % (time.time() - start))

if __name__ == '__main__':
    main()