
# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Data generators for translation data-sets."""

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

EOS = text_encoder.EOS_ID


_ENZU_TRAIN_DATASETS = [
    [
        "https://github.com/LauraMartinus/ukuxhumana/blob/master/clean/en_zu/en_zu.train.tar.gz?raw=true",
        (
            "enzu_parallel.train.en",
            "enzu_parallel.train.zu"
        )
    ]
]

_ENZU_TEST_DATASETS = [
    [
        "https://github.com/LauraMartinus/ukuxhumana/blob/master/clean/en_zu/en_zu.dev.tar.gz?raw=true",
        (
            "enzu_parallel.dev.notest.en",
            "enzu_parallel.dev.notest.zu"
        )
    ]
]

_ENZU_BPE_TRAIN_DATASETS = [
    [
        "http://github.com/LauraMartinus/ukuxhumana/blob/master/bpe/en_zu/en_zu.train.tar.gz?raw=true",
        (   
            "enzu_parallel.8000.train.en",
            "enzu_parallel.8000.train.zu"
        )

    ]
]

_ENZU_BPE_TEST_DATASETS = [
    [
        "http://github.com/LauraMartinus/ukuxhumana/blob/master/bpe/en_zu/en_zu.dev.tar.gz?raw=true",
        (   
            "enzu_parallel.8000.dev.en",
            "enzu_parallel.8000.dev.zu"
        )

    ]
]


@registry.register_problem
class TranslateEnzuRma(translate.TranslateProblem):
  """Problem spec for WMT English-Zulu translation."""
  def __init__(self,approx_vocab_size=32768):
        self.approx_vocab_size = approx_vocab_size

  @property
  def vocab_filename(self):
    return "vocab.enzn.%d" % self.approx_vocab_size


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENZU_TRAIN_DATASETS if train else _ENZU_TEST_DATASETS


@registry.register_problem
class TranslateEnzuBpeRma(translate.TranslateProblem):
  """Problem spec for WMT English-Zulu translation."""

  def __init__(self, bpe_tokens=8000):
    self.bpe_tokens=bpe_tokens

  @property
  def oov_token(self):
    return "<unk>"

  @property
  def vocab_filename(self):
    return "bpe.%d.vocab" % (self.bpe_tokens,)


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENZU_BPE_TRAIN_DATASETS if train else _ENZU_BPE_TEST_DATASETS


  def get_dataset(self, directory, filename, split):
    """Extract the EN_ZU corpus `filename` to directory unless it's there."""
    dataset = self.source_data_files(split)
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".zu") and
        tf.gfile.Exists(train_path + ".en")):
        corpus_file = generator_utils.maybe_download(
            directory, filename, dataset[0][0])
        
        with tarfile.open(corpus_file, "r:gz") as corpus_tar:
            corpus_tar.extractall(directory)
    return train_path

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the Autshomotso BPE en->zu task, training set."""
    # Get data
    filename = "enzu_parallel.%d.%s" % (self.bpe_vocab, dataset_split,)
    data_path = self.get_dataset(tmp_dir,filename,dataset_split)

    # Vocab
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      bpe_vocab = os.path.join(tmp_dir, self.vocab_filename)
      with tf.gfile.Open(bpe_vocab) as f:
        vocab_list = f.read().split("\n")
      vocab_list.append(self.oov_token)
      text_encoder.TokenTextEncoder(
          None, vocab_list=vocab_list).store_to_file(vocab_path)
    
    return text_problems.text2text_txt_iterator(data_path + ".en",
                                                data_path + ".zu")


