# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX IMDB preprocessing.

This file defines a template for TFX Transform component.
"""
import re
import string

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import constants

from models import features


def text_standardization(reviews, sequence_length=constants.SEQUENCE_LENGTH):
    """Preparing the text for training by filtering out unnecessary text"""
    # lowercase = tf.strings.lower(reviews)
    # sentences = tf.strings.regex_replace(lowercase, r" '| '|^'|'$", " ")
    # sentences = tf.strings.regex_replace(sentences, "[^a-z' ]", " ")
    
    tokenizer = Tokenizer(num_words=constants.VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(reviews)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')

    return padded

# TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}

    

    for key in features.FEATURE_KEYS:
        # Get the tokens for the dataset
        tokens = text_standardization(inputs[key])
        # Build a vocabulary for this feature.
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
                tokens,
                top_k=constants.VOCAB_SIZE,
                num_oov_buckets=constants.OOV_SIZE,
            )

    # Do not apply label transformation as it will result in wrong evaluation.
    outputs[features.transformed_name(features.LABEL_KEY)] = _fill_in_missing(inputs["label"], -1)

    return outputs

def _fill_in_missing(x, default_value):
  """Replace missing values in a SparseTensor.

  Fills in missing values of `x` with the default_value.

  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    default_value: the value with which to replace the missing values.

  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
