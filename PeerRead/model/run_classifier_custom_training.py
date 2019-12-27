# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

# pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
from modeling import model_training_utils
from nlp import bert_modeling as modeling
# from nlp import bert_models
from nlp import optimization
from nlp.bert import common_flags
from nlp.bert import tokenization
from nlp.bert import input_pipeline
from nlp.bert import model_saving_utils
from utils.misc import keras_utils
from utils.misc import tpu_lib
from hacking import bert_models

from PeerRead.dataset.dataset import make_input_fn_from_file, make_real_labeler

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only'],
    'One of {"train_and_eval", "export_only"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')

flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')

# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer(
    'max_seq_length', 250,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')

flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')
flags.DEFINE_string(
    'hub_module_url', None, 'TF-Hub path/url to Bert module. '
                            'If specified, init_checkpoint flag should not be used.')

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("seed", 0, "Seed for rng.")

common_flags.define_common_bert_flags()

# Data splitting details
flags.DEFINE_integer("num_splits", 10,
                     "number of splits")
flags.DEFINE_string("dev_splits", '11', "indices of development splits")
flags.DEFINE_string("test_splits", '11', "indices of test splits")

# Flags specifically related to PeerRead experiment

flags.DEFINE_string(
    "treatment", "theorem_referenced",
    "Covariate used as treatment."
)

flags.DEFINE_string("simulated", 'real', "whether to use real data ('real'), attribute based ('attribute'), "
                                         "or propensity score-based ('propensity') simulation"),
flags.DEFINE_float("beta0", 0.0, "param passed to simulated labeler, treatment strength")
flags.DEFINE_float("beta1", 0.0, "param passed to simulated labeler, confounding strength")
flags.DEFINE_float("gamma", 0.0, "param passed to simulated labeler, noise level")
flags.DEFINE_float("exogenous_confounding", 0.0, "amount of exogenous confounding in propensity based simulation")
flags.DEFINE_string("base_propensities_path", '', "path to .tsv file containing a 'propensity score' for each unit,"
                                                  "used for propensity score-based simulation")

flags.DEFINE_string("simulation_mode", 'simple', "simple, multiplicative, or interaction")

FLAGS = flags.FLAGS


def make_dragonnet_loss(binary_outcome):
    def dragonnet_loss(y_, t_, g, q0, q1, mask=None):

        y = tf.cast(y_, tf.float32)
        t = tf.cast(t_, tf.float32)

        if binary_outcome:
            q0_losses = -(y * tf.math.log(q0) + (1 - y) * tf.math.log(1. - q0)) * (1 - t)
            q1_losses = -(y * tf.math.log(q1) + (1 - y) * tf.math.log(1. - q1)) * t
        else:
            q0_losses = tf.square(y - q0) * (1 - t)
            q1_losses = tf.square(y - q1) * t

        g_losses = -(t * tf.math.log(g) + (1 - t) * tf.math.log(1. - g))

        net_losses = q0_losses + q1_losses + g_losses

        if mask:
            net_losses = net_losses[mask == 1]

        return tf.reduce_sum(net_losses)

    return dragonnet_loss


def train_input_fn():
    labeler = make_real_labeler(FLAGS.treatment, 'accepted')

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_input_fn = make_input_fn_from_file(
        input_files_or_glob=FLAGS.input_files,
        seq_length=FLAGS.max_seq_length,
        num_splits=1,
        dev_splits=[2],
        test_splits=[2],
        tokenizer=tokenizer,
        do_masking=False,
        is_training=True,
        is_pretraining=False,
        shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
        seed=FLAGS.seed,
        labeler=labeler)
    return train_input_fn(params={'batch_size': FLAGS.train_batch_size})


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.')

    # with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    #     input_meta_data = json.loads(reader.read().decode('utf-8'))

    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert20/'
    #
    # Configuration stuff
    #
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = FLAGS.num_train_epochs
    train_data_size = 12000
    steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
    warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
    initial_lr = FLAGS.learning_rate

    strategy = None
    if FLAGS.strategy_type == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    elif FLAGS.strategy_type == 'tpu':
        cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        raise ValueError('The distribution strategy type is not supported: %s' %
                         FLAGS.strategy_type)

    #
    # Modeling and training
    #

    # the model
    def _get_dragon_model():
        dragon_model, core_model = (
            bert_models.dragon_model(
                bert_config,
                max_seq_length=128,
                binary_outcome=True))
        dragon_model.optimizer = optimization.create_optimizer(
            initial_lr, steps_per_epoch * epochs, warmup_steps)
        return dragon_model, core_model

    # training utils expect loss_fn w/ this signature, where labels is second entry of dataset sample and
    # model_outputs = model(features)
    def loss_fn(labels, model_outputs):
        g, q0, q1 = model_outputs
        dragonnet_loss = make_dragonnet_loss(binary_outcome=True)
        loss = dragonnet_loss(labels['outcome'], labels['treatment'], g, q0, q1)
        return loss

    # input_data = train_input_fn()
    # train_iterator = iter(strategy.experimental_distribute_dataset(input_data))
    # inputs, _ = next(train_iterator)
    # dragon_model, core_model = _get_dragon_model()
    # model_outputs = dragon_model(inputs, training=True)

    # use TF-dev provided custom training utils to abstract away details about logging and training loop construction
    trained_model = model_training_utils.run_customized_training_loop(
        strategy=strategy,
        model_fn=_get_dragon_model,
        loss_fn=loss_fn,
        model_dir=FLAGS.model_dir,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=FLAGS.steps_per_loop,
        epochs=epochs,
        train_input_fn=train_input_fn,
        eval_input_fn=None,
        eval_steps=None,
        init_checkpoint=FLAGS.init_checkpoint,
        # metric_fn=metric_fn, #TODO: figure this out!
        metric_fn=None,
        custom_callbacks=None,
        run_eagerly=FLAGS.run_eagerly)

    if FLAGS.model_export_path:
        model_saving_utils.export_bert_model(
            FLAGS.model_export_path, model=trained_model)

    # # by-hand training loop for reference
    # dragon_model, _ = _get_dragon_model()
    # # the data
    # training_data = train_input_fn()
    #
    # for features, labels in training_data:
    #     with tf.GradientTape() as tape:
    #         g, q0, q1 = dragon_model(features)
    #         loss = dragonnet_loss(labels['outcome'], labels['treatment'], g, q0, q1)
    #
    #     grads = tape.gradient(loss, dragon_model.trainable_variables)
    #     dragon_model.optimizer.apply_gradients(zip(grads, dragon_model.trainable_variables))


if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)
