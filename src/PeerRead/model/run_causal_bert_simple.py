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

import os
import pandas as pd

from absl import app
from absl import flags
import tensorflow as tf

# pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
from src.tf_official.nlp import bert_modeling as modeling, optimization
# from nlp import bert_models
from src.tf_official.nlp.bert import tokenization, common_flags, model_saving_utils
from src.tf_official.utils.misc import tpu_lib
from src.causal_bert import bert_models

from src.PeerRead import make_input_fn_from_file, make_real_labeler

common_flags.define_common_bert_flags()

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

# Data splitting details
flags.DEFINE_integer("num_splits", 10,
                     "number of splits")
flags.DEFINE_string("dev_splits", '9', "indices of development splits")
flags.DEFINE_string("test_splits", '9', "indices of test splits")

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

flags.DEFINE_string("prediction_file", "../output/predictions.tsv", "path where predictions (tsv) will be written")

FLAGS = flags.FLAGS


def make_dataset(is_training: bool):
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
        is_training=is_training,
        shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
        seed=FLAGS.seed,
        labeler=labeler)

    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size

    return train_input_fn(params={'batch_size': batch_size})


def make_dragonnet_losses():
    @tf.function
    def q0_loss(yt, q0):
        yt = tf.cast(yt, tf.float32)
        y, t = tf.unstack(yt, axis=-1, num=2)

        q0_losses = -(y * tf.math.log(q0) + (1 - y) * tf.math.log(1. - q0)) * (1 - t)
        return tf.reduce_sum(q0_losses)

    @tf.function
    def q1_loss(yt, q1):
        yt = tf.cast(yt, tf.float32)
        y, t = tf.unstack(yt, axis=-1, num=2)

        q1_losses = -(y * tf.math.log(q1) + (1 - y) * tf.math.log(1. - q1)) * t
        return tf.reduce_sum(q1_losses)

    @tf.function
    def g_loss(t, g):
        t = tf.cast(t, tf.float32)
        g_losses = -(t * tf.math.log(g) + (1 - t) * tf.math.log(1. - g))
        return tf.reduce_sum(g_losses)

    return g_loss, q0_loss, q1_loss


def make_yt_metric(metric, name, flip_t=False):
    class YTMetric(metric):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def update_state(self, yt, y_pred, **kwargs):
            yt = tf.cast(yt, tf.float32)
            y, t = tf.unstack(yt, axis=-1, num=2)

            y = tf.expand_dims(y, 1)  # to match y_pred
            if flip_t:
                t = 1 - t

            super().update_state(y, y_pred, sample_weight=t)

    return YTMetric(name=name)


def make_dragonnet_metrics():
    METRICS = [
        tf.keras.metrics.TruePositives,
        tf.keras.metrics.FalsePositives,
        tf.keras.metrics.TrueNegatives,
        tf.keras.metrics.FalseNegatives,
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.AUC
    ]

    NAMES = ['true_positive', 'false_positive', 'true_negative', 'false_negative',
             'binary_accuracy', 'precision', 'recall', 'auc']

    q0_names = ['q0/' + n for n in NAMES]
    q0_metrics = [make_yt_metric(m, name=n, flip_t=True) for m, n in zip(METRICS, q0_names)]

    q1_names = ['q1/' + n for n in NAMES]
    q1_metrics = [make_yt_metric(m, name=n, flip_t=False) for m, n in zip(METRICS, q1_names)]

    g_names = ['g/' + n for n in NAMES]
    g_metrics = [m(name=n) for m, n in zip(METRICS, g_names)]

    return {'g': g_metrics, 'q0': q0_metrics, 'q1': q1_metrics}


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
            bert_models.dragon_model_simple(
                bert_config,
                max_seq_length=250,
                binary_outcome=True))
        dragon_model.optimizer = optimization.create_optimizer(
            initial_lr, steps_per_epoch * epochs, warmup_steps)
        return dragon_model, core_model

    # we'll need a hack to let keras loss depend on multiple labels. Which is just plain stupid design.
    @tf.function
    def _keras_format(features, labels):
        # features, labels = sample
        y = labels['outcome']
        t = labels['treatment']
        yt = tf.stack([y, t], axis=-1)
        labels = {'g': labels['treatment'], 'q0': yt, 'q1': yt}
        return features, labels

    # losses
    g_loss, q0_loss, q1_loss = make_dragonnet_losses()

    # training. strategy.scope context allows use of multiple devices
    with strategy.scope():
        input_data = make_dataset(is_training=True)
        keras_train_data = input_data.map(_keras_format)

        dragon_model, core_model = _get_dragon_model()
        optimizer = dragon_model.optimizer

        if FLAGS.init_checkpoint:
            checkpoint = tf.train.Checkpoint(model=core_model)
            checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()

        dragon_model.compile(optimizer=optimizer,
                             loss={'g': g_loss, 'q0': q0_loss, 'q1': q1_loss},
                             loss_weights={'g': 0.8, 'q0': 0.1, 'q1': 0.1},
                             metrics=make_dragonnet_metrics())

        summary_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir, update_freq=640)
        checkpoint_dir = os.path.join(FLAGS.model_dir, 'model_checkpoint.{epoch:02d}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)

        callbacks = [summary_callback, checkpoint_callback]

        dragon_model.fit(
            x=keras_train_data,
            # validation_data=evaluation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            # vailidation_steps=eval_steps,
            callbacks=callbacks)

    # save the model
    if FLAGS.model_export_path:
        model_saving_utils.export_bert_model(
            FLAGS.model_export_path, model=dragon_model)

    # make predictions and write to file
    # NOTE: theory suggests we should make predictions on heldout data ("cross fitting" or "sample splitting")
    # but our experiments showed best results by just reusing the data
    # You can accomodate sample splitting by using the splitting arguments for the dataset creation

    eval_data = make_dataset(is_training=False)

    with tf.io.gfile.GFile(FLAGS.prediction_file, "w") as writer:
        attribute_names = ['in_test',
                           'treatment_probability',
                           'expected_outcome_st_treatment', 'expected_outcome_st_no_treatment',
                           'outcome', 'treatment',
                           'index']
        header = "\t".join(
            attribute_name for attribute_name in attribute_names) + "\n"
        writer.write(header)

        for f, l in eval_data:
            g_pred, q0_pred, q1_pred = dragon_model.predict(f)
            attributes = [l['in_test'].numpy(),
                          g_pred, q1_pred, q0_pred,
                          l['outcome'].numpy(), l['treatment'].numpy(), l['index'].numpy()]
            at_df = pd.DataFrame(attributes).T
            writer.write(at_df.to_csv(sep="\t", header=False))


if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)
