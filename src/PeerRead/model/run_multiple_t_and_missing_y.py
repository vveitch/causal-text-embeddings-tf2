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
from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert import tokenization, common_flags
from tf_official.utils.misc import tpu_lib
from causal_bert import bert_models

from causal_bert.data_utils import load_basic_bert_data, dataset_labels_from_pandas, add_masking, dataset_to_pandas_df

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
flags.DEFINE_string('label_df_file', None,
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
        def _hydra_keras_format(features, labels):
            # construct the label dictionary
            y = tf.cast(labels['outcome'], tf.int32)
            y_is_obs_label = tf.cast(labels['outcome_observed'], tf.int32)
            labels_out = {'g0': labels['treatment'], 'g1': labels['treatment'], 'y_is_obs': y_is_obs_label}
            for treat in range(num_treatments):
                labels_out[f"q{treat}"] = y

            # construct the sample weighting dictionary
            t = labels['treatment']
            y_is_obs_weight = tf.cast(labels['outcome_observed'], tf.float32)[:, 0]
            y_is_obs_bool = tf.cast(labels['outcome_observed'], tf.bool)

            sample_weights = {'g0': 1 - y_is_obs_weight,
                              'g1': y_is_obs_weight}  # these heads correspond to P(T| missing = ~, x)
            # mask a treatment head if (1) that treatment wasn't assigned, or (2) the outcome is missing
            for treat in range(num_treatments):
                treat_active = tf.equal(t, treat)
                treat_active = tf.logical_and(treat_active, y_is_obs_bool)[:, 0]
                sample_weights[f"q{treat}"] = tf.cast(treat_active, tf.float32)
            return features, labels_out, sample_weights

    return _hydra_keras_format


def make_dataset(tf_record_files: str, is_training: bool, num_treatments: int, missing_outcomes=False, do_masking=False,
                 input_pipeline_context=None):
    df_file = FLAGS.label_df_file
    dataset = load_basic_bert_data(tf_record_files, FLAGS.max_seq_length, is_training=is_training,
                                   input_pipeline_context=input_pipeline_context)

    label_df = pd.read_feather(df_file)
    dataset = dataset_labels_from_pandas(dataset, label_df)

    # todo: hardcoded for demo, but not the smartest way to do this
    def _standardize_label_naming(f, l):
        l['outcome'] = l.pop('accepted')
        l['treatment'] = l.pop('year')
        if missing_outcomes:
            l['outcome_observed'] = tf.not_equal(l['outcome'], -1)
        # placeholder so that passed in labels are non-negative
        l['outcome'] = tf.where(l['outcome_observed'], l['outcome'], tf.zeros_like(l['outcome']))

        return f, l

    dataset = dataset.map(_standardize_label_naming, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.cache()

    if do_masking:
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        dataset = add_masking(dataset, tokenizer=tokenizer)

    if is_training:
        # batching needs to happen before sample weights are created
        dataset = dataset.shuffle(25000)
        dataset = dataset.batch(FLAGS.train_batch_size, drop_remainder=True)

        # create sample weights and label outputs in the manner expected by keras
        hydra_keras_format = make_hydra_keras_format(num_treatments, missing_outcomes=missing_outcomes)
        dataset = dataset.map(hydra_keras_format, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    else:
        return dataset.batch(FLAGS.eval_batch_size)


def make_hydra_metrics(num_treatments, missing_outcomes=False):
    basic_metrics = [
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.AUC
    ]

    q_names = ['/binary_accuracy', '/precision', '/recall', '/auc']

    if missing_outcomes:
        metrics = {'g0': [tf.keras.metrics.SparseCategoricalAccuracy()],
                   'g1': [tf.keras.metrics.SparseCategoricalAccuracy()],
                   'y_is_obs': [tf.keras.metrics.BinaryAccuracy()]}
    else:
        metrics = {'g': [tf.keras.metrics.SparseCategoricalAccuracy()]}

    for treat in range(num_treatments):
        q_metric = [m(name=n) for m, n in zip(basic_metrics, q_names)]
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
    train_data_size = 11778  # todo: fix hardcording
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
        hydra_model, core_model = (
            bert_models.hydra_model(
                bert_config,
                max_seq_length=FLAGS.max_seq_length,
                binary_outcome=True,
                num_treatments=num_treatments,
                missing_outcomes=missing_outcomes,
                use_unsup=do_masking,
                max_predictions_per_seq=20,
                unsup_scale=1.))

        # WARNING: the original optimizer causes a bug where loss increases after first epoch
        # hydra_model.optimizer = optimization.create_optimizer(
        #     FLAGS.train_batch_size * initial_lr, steps_per_epoch * epochs, warmup_steps)
        hydra_model.optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.train_batch_size * initial_lr)

        return hydra_model, core_model

    # training. strategy.scope context allows use of multiple devices
    with strategy.scope():
        train_data = make_dataset(tf_record_files=FLAGS.input_files,
                                  is_training=True,
                                  num_treatments=num_treatments, missing_outcomes=missing_outcomes,
                                  do_masking=FLAGS.do_masking)
        hydra_model, core_model = _get_hydra_model(FLAGS.do_masking)
        optimizer = hydra_model.optimizer
        print(hydra_model.summary())

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

        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
        if latest_checkpoint:
            hydra_model.load_weights(latest_checkpoint)

        hydra_model.compile(optimizer=optimizer,
                            loss=losses,
                            loss_weights=loss_weights,
                            weighted_metrics=make_hydra_metrics(num_treatments, missing_outcomes))

        summary_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir, update_freq=128)
        checkpoint_dir = os.path.join(FLAGS.model_dir, 'model_checkpoint.{epoch:02d}')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True)

        callbacks = [summary_callback, checkpoint_callback]

        hydra_model.fit(
            x=train_data,
            # validation_data=evaluation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            # validation_steps=eval_steps,
            callbacks=callbacks)

    # save a final model checkpoint (so we can restore weights into model w/o training idiosyncracies)
    hydra_model.optimizer = None
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
    hydra_model.compile()  # seems to erratically cause bugs to omit this? very puzzling

    outputs = hydra_model.predict(x=eval_data)

    out_dict = {}
    if missing_outcomes:

        for t, g0 in enumerate(tf.unstack(outputs[0], axis=-1)):
            out_dict['g0_' + str(t)] = g0.numpy()

        for t, g1 in enumerate(tf.unstack(outputs[1], axis=-1)):
            out_dict['g1_' + str(t)] = g1.numpy()

        out_dict['prob_y_obs'] = np.squeeze(outputs[2])

        for out, q in enumerate(outputs[3:]):
            out_dict['q' + str(out)] = np.squeeze(q)

    else:

        for t, g in enumerate(tf.unstack(outputs[0], axis=-1)):
            out_dict['g_' + str(t)] = g.numpy()

        for out, q in enumerate(outputs[1:]):
            out_dict['q' + str(out)] = np.squeeze(q)

    predictions = pd.DataFrame(out_dict)

    label_dataset = eval_data.map(lambda f, l: l)
    data_df = dataset_to_pandas_df(label_dataset)

    outs = data_df.join(predictions)
    with tf.io.gfile.GFile(FLAGS.prediction_file, "w") as writer:
        writer.write(outs.to_csv(sep="\t"))


if __name__ == '__main__':
    flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)
