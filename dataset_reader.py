import os.path
import collections
import codecs
import _pickle as cPickle

class DatasetReader:

  """Reads the datasets (train data, test data, etc)"""

  def __init__(self, args, train=True):
    self.save_dir = args.save_dir # models dir
    self.batch_size = args.batch_size
    self.num_steps = args.num_steps
    self.save_vocab_file = os.path.join(self.save_dir, "vocab.pkl")
    self.vocab_file = args.vocab_file

    self.wrd2idx = None
    self.vocab_size = None

    if train:
      self.train_data = self.read_dataset(args.train_file)
      self.dev_data = self.read_dataset(args.dev_file)
      self.read_or_build_vocab() # read vocab or build if needed
    else:
      self.test_data = self.read_dataset(args.test_file)
      self.load_vocab() # load vocab

  def read_dataset(self, file_name):
    """Read dataset from file

    :param file_name: A string, name of the dataset file
    :return: A list of word tokens
    """

    if not file_name or not os.path.exists(file_name):
      return None

    start_token = self.get_start_token()
    end_token = self.get_end_token()
    data = [start_token]
    with open(file_name, 'r') as f:
      for line in f:
        line = line.strip().lower().split()
        for token in line:
          data.append(token)
        data.append(end_token)
    return data

  def read_vocab(self, file_name):
    """Read vocabulary file

    :param file_name: A string, vocabulary filename
    :return: A dict, mapping from words to idx
    """

    wrd2idx = {}
    unk_token = self.get_unk_token()
    with open(file_name, 'r') as f:
      for wrd in f:
        wrd2idx[wrd.strip()] = len(wrd2idx)
      if unk_token not in wrd2idx:
        wrd2idx[unk_token] = len(wrd2idx)
    return wrd2idx

  def build_vocab(self, train_data):
    """Build vocabulary from training corpus in case we don't have it

    NOTE that this is not used for now... As I know it is usually used
    for considering only the top k frequent words and the rest are mapped
    to unk

    :param train_data: training corpus
    :return: A dict and set, a mapping from word to id and set of unk words
    """

    counter = collections.Counter(train_data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[-1], x[0]))

    words, _ = list(zip(*count_pairs))
    wrd2idx = dict(zip(words, range(len(words))))

    unk_token = self.get_unk_token()

    if unk_token not in wrd2idx:
      wrd2idx[unk_token] = len(wrd2idx)

    return wrd2idx

  def read_or_build_vocab(self):
    """Preprocess the dataset and build vocabulary"""

    if not self.vocab_file or not os.path.exists(self.vocab_file):
      self.wrd2idx = self.build_vocab(self.train_data)
    else:
      self.wrd2idx = self.read_vocab(self.vocab_file)

    self.vocab_size = len(self.wrd2idx)
    with codecs.open(self.save_vocab_file, 'wb') as f:
      cPickle.dump(self.wrd2idx, f)

  def load_vocab(self):
    """Load vocabulary"""

    with codecs.open(self.save_vocab_file, 'rb') as f:
      self.wrd2idx = cPickle.load(f)
      self.vocab_size = len(self.wrd2idx)

  def map_data_to_idx(self, data):
    """Map data tokens to ids

    :param data: A list of tokens
    :return: A dict mapping tokens to ids
    """

    idxs = []
    for token in data:
      token = token.lower()
      if token in self.wrd2idx:
        idxs.append(self.wrd2idx[token])
      else:
        idxs.append(self.wrd2idx[self.get_unk_token()])
    return idxs

  def data_iterator(self, raw_data, batch_size, num_steps):
    """Construct and return a data iterator

    :param raw_data: A list representing data
    :param batch_size: An integer, batch size
    :param num_steps: An integer, number of time steps
    :return: A data iterator
    """

    data_len = len(raw_data)
    batch_len = data_len // batch_size

    data = []
    for i in range(batch_size):
      x = raw_data[batch_len * i : batch_len * (i+1)]
      data.append(x)

    epoch_size = (batch_len - 1) // num_steps # /(B*T)

    assert epoch_size > 0, 'epoch size is 0, decrease batch_size or num_steps'

    # construct input and output sequences
    # e.g suppose we have w1, w2, w3, w4 sequence and let T = 2
    # x = [w1, w2], and y[w2, w3] ...
    # so if w1 is the input we expect to get w2 to model p(w2 | w1)

    for i in range(epoch_size):
      x_seq = list()
      y_seq = list()
      for j in range(batch_size):
        x = data[j][i*num_steps:(i+1)*num_steps]
        y = data[j][i*num_steps+1:(i+1)*num_steps+1]
        x_seq.append(self.map_data_to_idx(x))
        y_seq.append(self.map_data_to_idx(y))
      yield (x_seq, y_seq) # this will return a generator object

  @staticmethod
  def get_start_token():
    return '<s>'

  @staticmethod
  def get_end_token():
    return '</s>'

  @staticmethod
  def get_unk_token():
    return '<unk>'


