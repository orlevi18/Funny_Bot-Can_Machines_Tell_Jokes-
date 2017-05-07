# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import json
import tensorflow as tf

import nltk
import itertools

def _read_words(filename):
  all_text=[]
  with open(filename, "r") as f:

      # f.read().decode("utf-8").replace("\n", "<eos>").split()
    data = json.load(f)
    for joke in data:
        # convert to lowercase
        
        if 'title' in data:
            text=joke['title']+" . "+joke['body']
        else:
            text=joke['body']
        sentences = itertools.chain(*[nltk.sent_tokenize(text.lower())])
        sentences = [x for x in sentences]
        # Append JOKE_START and JOKE_END
        sentences = ["SOJ"]+sentences
        sentences.append("EOJ")
        sentences=' '.join(sentences)
        all_text.append(sentences)

    # remove punctuation
    #chars_to_remove = ['\"','\"', '\'', ',']
    #for c in chars_to_remove:
    #    data=data.replace(c,'')
        
    all_text=' '.join(all_text)
    all_text = nltk.word_tokenize(all_text)   
    return all_text


def _build_vocab(filename,vocab_size):

  data = _read_words(filename)
  counter = collections.Counter(data)
  print(sum(1 for i in counter.values() if i>3))
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # limit vocabulary to X most frequent words
  count_pairs = count_pairs[0:vocab_size-1]
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  # add unk marker for out of vocabulary words
  word_to_id["UNK"]=len(words)
  
  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  # return index of unk marker for of vocabulary words
  data = _read_words(filename)
  return [word_to_id[word] if word in word_to_id else word_to_id["UNK"] for word in data]


def ptb_raw_data(vocab_size,data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "wocka_train.json")
  valid_path = os.path.join(data_path, "wocka_valid.json")
  test_path = os.path.join(data_path, "wocka_test.json")

  word_to_id = _build_vocab(train_path,vocab_size)
  train_data = _file_to_word_ids(train_path,word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)

  return train_data,valid_data,test_data,word_to_id


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps], tf.ones_like([0, i * num_steps]))
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1], tf.ones_like([0, i * num_steps + 1]))
    y.set_shape([batch_size, num_steps])
    return x, y