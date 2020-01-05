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
from tf_official.nlp import bert_modeling as modeling, optimization
from tf_official.nlp.bert import tokenization, common_flags, model_saving_utils
from tf_official.utils.misc import tpu_lib
from causal_bert import bert_models

from causal_bert.data_utils import load_basic_bert_data, dataset_labels_from_pandas, add_masking

common_flags.define_common_bert_flags()

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only'],
    'One of {"train_and_eval", "export_only"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`.')

flags.DEFINE_bool(
    "do_masking", True,
    "Whether to randomly mask input words during training (serves as a sort of regularization)")

flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')
flags.DEFINE_string('label_df_file', 'dat/PeerRead/proc/arxiv-all-multi-treat-and-missing-outcomes.feather',
                    'File path for pandas dataframe containing labels')

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

flags.DEFINE_string("prediction_file", "out/predictions.tsv", "path where predictions (tsv) will be written")

# more complicated data
flags.DEFINE_integer("num_treatments", 11, "number of treatment levels")
flags.DEFINE_bool("missing_outcomes", True, "Whether there are missing outcomes")

FLAGS = flags.FLAGS


def make_hydra_keras_format(num_treatments, missing_outcomes=False):
    if not missing_outcomes:
        @tf.function
        def _hydra_keras_format(features, labels):
            y = labels['outcome']
            t = tf.cast(labels['treatment'], tf.float32)
            labels = {'g': labels['treatment']}
            sample_weights = {}
            for treat in range(num_treatments):
                labels[f"q{treat}"] = y
                treat_active = tf.equal(t, treat)
                sample_weights[f"q{treat}"] = tf.cast(treat_active, tf.float32)
            return features, labels, sample_weights

    elif missing_outcomes:
        @tf.function
        def _hydra_keras_format(features, labels):
            # construct the label dictionary
            y = labels['outcome']
            y_is_obs_label = tf.cast(labels['outcome_observed'], tf.int32)
            labels_out = {'g0': labels['treatment'], 'g1': labels['treatment'], 'y_is_obs': y_is_obs_label}
            for treat in range(num_treatments):
                labels_out[f"q{treat}"] = y

            # construct the sample weighting dictionary
            t = labels['treatment']
            y_is_obs = tf.cast(labels['outcome_observed'], tf.float32)[:, 0]

            sample_weights = {'g0': 1 - y_is_obs, 'g1': y_is_obs}  # these heads correspond to P(T| missing = ~, x)
            # mask a treatment head if (1) that treatment wasn't assigned, or (2) the outcome is missing
            for treat in range(num_treatments):
                treat_active = tf.equal(t, treat)
                treat_active = tf.logical_and(treat_active, labels['outcome_observed'])[:, 0]
                sample_weights[f"q{treat}"] = tf.cast(treat_active, tf.float32)
            return features, labels_out, sample_weights

    return _hydra_keras_format


def make_dataset(tf_record_files: str, is_training: bool, num_treatments: int, missing_outcomes=False, do_masking=False,
                 input_pipeline_context=None):
    df_file = FLAGS.label_df_file
    dataset = load_basic_bert_data(tf_record_files, 250, is_training=is_training,
                                   input_pipeline_context=input_pipeline_context)

    if do_masking:
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        dataset = add_masking(dataset, tokenizer=tokenizer)

    label_df = pd.read_feather(df_file)
    dataset = dataset_labels_from_pandas(dataset, label_df)

    # todo: hardcoded for demo, but not the smartest way to do this
    def _standardize_label_naming(f, l):
        l['outcome'] = l.pop('accepted')
        l['treatment'] = l.pop('year')
        if missing_outcomes:
            l['outcome_observed'] = tf.not_equal(l['outcome'], -1)
        return f, l

    dataset = dataset.map(_standardize_label_naming)

    if is_training:
        # batching needs to happen before sample weights are created
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(FLAGS.train_batch_size, drop_remainder=True)
        dataset = dataset.prefetch(1024)

        # create sample weights and label outputs in the manner expected keras
        hydra_keras_format = make_hydra_keras_format(num_treatments, missing_outcomes=missing_outcomes)
        dataset = dataset.map(hydra_keras_format)

        return dataset

    else:
        return dataset.batch(FLAGS.eval_batch_size)


def make_hydra_metrics(num_treatments, missing_outcomes=False):
    METRICS = [
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.AUC
    ]

    NAMES = ['binary_accuracy', 'precision', 'recall', 'auc']

    if missing_outcomes:
        metrics = {'g0': [tf.keras.metrics.SparseCategoricalAccuracy()],
                   'g1': [tf.keras.metrics.SparseCategoricalAccuracy()],
                   'y_is_obs': [tf.keras.metrics.BinaryAccuracy()]}
    else:
        metrics = {'g': [tf.keras.metrics.SparseCategoricalAccuracy()]}

    for treat in range(num_treatments):
        q_metric = [m(name=n) for m, n in zip(METRICS, NAMES)]
        metrics[f"q{treat}"] = q_metric

    return metrics


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
    train_data_size = 50000  # todo: fix hardcording
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

    num_treatments = FLAGS.num_treatments
    missing_outcomes = FLAGS.missing_outcomes

    # the model
    def _get_hydra_model(do_masking):
        dragon_model, core_model = (
            bert_models.hydra_model(
                bert_config,
                max_seq_length=FLAGS.max_seq_length,
                binary_outcome=True,
                num_treatments=num_treatments,
                missing_outcomes=missing_outcomes,
                use_unsup=do_masking,
                max_predictions_per_seq=20,
                unsup_scale=1.))

        dragon_model.optimizer = optimization.create_optimizer(
            FLAGS.train_batch_size * initial_lr, steps_per_epoch * epochs, warmup_steps)
        return dragon_model, core_model

    # training. strategy.scope context allows use of multiple devices
    with strategy.scope():
        train_data = make_dataset(tf_record_files=FLAGS.input_files,
                                  is_training=True,
                                  num_treatments=num_treatments, missing_outcomes=missing_outcomes,
                                  do_masking=FLAGS.do_masking)
        hydra_model, core_model = _get_hydra_model(FLAGS.do_masking)
        optimizer = hydra_model.optimizer

        if FLAGS.init_checkpoint:
            checkpoint = tf.train.Checkpoint(model=core_model)
            checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()

        if not missing_outcomes:
            losses = {'g': 'sparse_categorical_crossentropy'}
            loss_weights = {'g': 1.0}
        else:
            losses = {'g0': 'sparse_categorical_crossentropy', 'g1': 'sparse_categorical_crossentropy',
                      'y_is_obs': 'binary_crossentropy'}
            loss_weights = {'g0': 1.0, 'g1': 1.0, 'y_is_obs': 1.0}

        for treat in range(num_treatments):
            losses[f"q{treat}"] = 'binary_crossentropy'
            loss_weights[f"q{treat}"] = 0.1

        hydra_model.compile(optimizer=optimizer,
                            loss=losses,
                            loss_weights=loss_weights,
                            weighted_metrics=make_hydra_metrics(num_treatments, missing_outcomes))

        summary_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir, update_freq=128)
        checkpoint_dir = os.path.join(FLAGS.model_dir, 'model_checkpoint.{epoch:02d}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)

        callbacks = [summary_callback, checkpoint_callback]

        hydra_model.fit(
            x=train_data,
            # validation_data=evaluation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            # vailidation_steps=eval_steps,
            callbacks=callbacks)

    # save a final model checkpoint (so we can restore weights into model w/o training idiosyncracies)
    if FLAGS.model_export_path:
        model_export_path = FLAGS.model_export_path
    else:
        model_export_path = os.path.join(FLAGS.model_dir, 'trained/hydra.ckpt')

    checkpoint = tf.train.Checkpoint(model=hydra_model)
    saved_path = checkpoint.save(model_export_path)

    # make predictions and write to file
    # NOTE: theory suggests we should make predictions on heldout data ("cross fitting" or "sample splitting")
    # but our experiments showed best results by just reusing the data
    # You can accommodate sample splitting by using the splitting arguments for the dataset creation

    eval_data = make_dataset(FLAGS.input_files, is_training=False, do_masking=False,
                             num_treatments=num_treatments, missing_outcomes=missing_outcomes)
    hydra_model, core_model = _get_hydra_model(do_masking=False)
    checkpoint = tf.train.Checkpoint(model=hydra_model)
    checkpoint.restore(saved_path).assert_existing_objects_matched()
    hydra_model.compile()

    with tf.io.gfile.GFile(FLAGS.prediction_file, "w") as writer:
        names = hydra_model.output_names
        if missing_outcomes:
            g0_names = ['g0_' + str(t) for t in range(num_treatments)]
            g1_names = ['g1_' + str(t) for t in range(num_treatments)]
            names = g0_names + g1_names + names[2:]
        else:
            g_names = ['g' + str(t) for t in range(num_treatments)]
            names = g_names + names[1:]

        names = ['id', 'outcome', 'treatment'] + names
        header = "\t".join(name for name in names) + "\n"
        writer.write(header)

        for f, l in eval_data:
            outputs = hydra_model.predict(f)
            if missing_outcomes:
                g0s = [g.numpy() for g in tf.unstack(outputs[0])]
                g1s = [g.numpy() for g in tf.unstack(outputs[1])]
                m = [outputs[2].numpy()]
                qs = [q.numpy() for q in outputs[3:]]
                predictions = g0s + g1s + m + qs
            else:
                gs = [g.numpy() for g in tf.unstack(outputs[0])]
                qs = [q.numpy() for q in outputs[1:]]
                predictions = gs + qs

            labels = [l['id'], l['outcome'].numpy(), l['treatment'].numpy()]  # treatments is sparse coded

            outs = pd.DataFrame(labels + predictions).T
            writer.write(outs.to_csv(sep="\t", header=False))


if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)
