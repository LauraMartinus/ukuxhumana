
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
        "https://github.com/LauraMartinus/ukuxhumana/blob/master/bpe/en_zu/en_zu.train.tar.gz?raw=true",
        (   
            "enzu_bpe_parallel.train.en",
            "enzu_bpe_parallel.train.zu"
        )

    ]
]

_ENZU_BPE_TEST_DATASETS = [
    [
        "https://github.com/LauraMartinus/ukuxhumana/blob/master/bpe/en_zu/en_zu.dev.tar.gz?raw=true",
        (   
            "enzu_bpe_parallel.dev.en",
            "enzu_bpe_parallel.dev.zu"
        )

    ]
]


@registry.register_problem
class TranslateEnzuRma(translate.TranslateProblem):
  """Problem spec for WMT English-Zulu translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_filename(self):
    return "vocab.enzn.%d" % self.approx_vocab_size


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENZU_TRAIN_DATASETS if train else _ENZU_TEST_DATASETS


@registry.register_problem
class TranslateEnzuBpeRma(translate.TranslateProblem):
  """Problem spec for WMT English-Zulu translation."""

  @property
  def oov_token(self):
    return "UNK"

  @property
  def vocab_filename(self):
    return "vocab.quoted.bpe.40000"


  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENZU_BPE_TRAIN_DATASETS if train else _ENZU_BPE_TEST_DATASETS

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Instance of token generator for the Autshomotso BPE en->zu task, training set."""
    datasets = self.source_data_files(dataset_split)
    
    # Vocab
    vocab_path = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_path):
      bpe_vocab = os.path.join(tmp_dir, "vocab.quoted.bpe.40000")
      with tf.gfile.Open(bpe_vocab) as f:
        vocab_list = f.read().split("\n")
      vocab_list.append(self.oov_token)
      text_encoder.TokenTextEncoder(
          None, vocab_list=vocab_list).store_to_file(vocab_path)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = translate.compile_data(tmp_dir, datasets, "%s-compiled-%s" % (self.name,
                                                                    tag))
    return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                data_path + ".lang2")


