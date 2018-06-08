# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf

from .loss import Loss


class CrossEntropyLoss(Loss):
  def __init__(self, params, model, name="cross_entropy_loss"):
    super(CrossEntropyLoss, self).__init__(params, model, name)

  def _compute_loss(self, input_dict):
    logits = input_dict['decoder_output']['logits']
    labels = input_dict['target_tensors'][0]
    has_nan = tf.logical_or(tf.reduce_any(tf.is_nan(logits)),
                            tf.is_inf(tf.reduce_max(tf.abs(logits))))
    with tf.control_dependencies([tf.Assert(tf.equal(has_nan, False), [logits],
                                            summarize=100000,
                                            name="nan_in_logits")]):
      loss = tf.losses.softmax_cross_entropy(logits=logits,
                                             onehot_labels=labels)
    return loss
