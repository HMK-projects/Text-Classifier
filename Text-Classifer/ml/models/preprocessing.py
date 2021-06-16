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
"""TFX penguin preprocessing.

This file defines a template for TFX Transform component.
"""
from typing import Text
import re
import string

import tensorflow as tf
import tensorflow_transform as tft
from models import constants

from models import features


def text_standardization(input_data: Text) -> Text:
    """Preparing the text for training by filtering out unnecessary text"""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                   '[%s]' % re.escape(string.punctuation),
                                   '')

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
        # Build a vocabulary for this feature.
        outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
                text_standardization(inputs[key])
            )
        print('outputs:', outputs[features.transformed_name(key)])

    # Do not apply label transformation as it will result in wrong evaluation.
    outputs[features.transformed_name(features.LABEL_KEY)] = inputs[
        features.LABEL_KEY
    ]

    return outputs
