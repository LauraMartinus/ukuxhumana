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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

_ENTN_TRAIN_DATASETS = [
  [
    "",
    (
      "data/eng_tswane/entn_parallel.train.en",
      "data/eng_tswane/entn_parallel.train.tn"
    )
  ]
]

_ENTN_TEST_DATASETS = [
  [
    "",
    (
      "data/eng_tswane/entn_parallel.dev.en",
      "data/eng_tswane/entn_parallel.dev.tn"
    )
  ]
]


@registry.register_problem
class TranslateEnTnRMA(translate.TranslateProblem):
  """Problem spec for WMT English-Tswane translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENTN_TRAIN_DATASETS if train else _ENTN_TEST_DATASETS


