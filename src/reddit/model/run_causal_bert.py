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
import time

import numpy as np
import pandas as pd

from absl import app
from absl import flags
import tensorflow as tf

from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert import tokenization, common_flags
from tf_official.utils.misc import tpu_lib
from causal_bert import bert_models
from causal_bert.data_utils import dataset_to_pandas_df, filter_training

from reddit.dataset.dataset import make_input_fn_from_file, make_real_labeler, make_subreddit_based_simulated_labeler

common_flags.define_common_bert_flags()

flags.DEFINE_enum(
    'mode', 'train_and_predict', ['train_only', 'train_and_predict', 'predict_only'],
    'One of {"train_and_predict", "predict_only"}. `train_and_predict`: '
    'trains the model and make predictions. '
    '`predict_only`: loads a trained model and makes predictions.')

flags.DEFINE_string('saved_path', None,
                    'Relevant only if mode is predict_only. Path to pre-trained model')

flags.DEFINE_bool(
    "do_masking", True,
    "Whether to randomly mask input words during training (serves as a sort of regularization)")

flags.DEFINE_float("treatment_loss_weight", 1.0, "how to weight the treatment prediction term in the loss")


flags.DEFINE_bool(
    "fixed_feature_baseline", False,
    "Whether to use BERT to produced fixed features (no finetuning)")


flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')

# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')

flags.DEFINE_integer('train_batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 64, 'Batch size for evaluation.')
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
flags.DEFINE_string("subreddits", '13,6,8', "the list of subreddits to train on")
flags.DEFINE_bool("use_subreddit", False, "whether to use the subreddit index as a feature")
flags.DEFINE_string("simulated", 'real', "whether to use real data ('real'), attribute based ('attribute'), "
                                         "or propensity score-based ('propensity') simulation"),
flags.DEFINE_float("beta0", 0.25, "param passed to simulated labeler, treatment strength")
flags.DEFINE_float("beta1", 0.0, "param passed to simulated labeler, confounding strength")
flags.DEFINE_float("gamma", 0.0, "param passed to simulated labeler, noise level")
flags.DEFINE_float("exogenous_confounding", 0.0, "amount of exogenous confounding in propensity based simulation")
flags.DEFINE_string("base_propensities_path", '', "path to .tsv file containing a 'propensity score' for each unit,"
                                                  "used for propensity score-based simulation")

flags.DEFINE_string("simulation_mode", 'simple', "simple, multiplicative, or interaction")

flags.DEFINE_string("prediction_file", "../output/predictions.tsv", "path where predictions (tsv) will be written")

FLAGS = flags.FLAGS


def _keras_format(features, labels):
    # features, labels = sample
    y = labels['outcome']
    t = tf.cast(labels['treatment'], tf.float32)
    labels = {'g': labels['treatment'], 'q0': y, 'q1': y}
    sample_weights = {'q0': 1 - t, 'q1': t}
    return features, labels, sample_weights


def make_dataset(is_training: bool, do_masking=False):
    if FLAGS.simulated == 'real':
        labeler = make_real_labeler(FLAGS.treatment, 'log_score')

    elif FLAGS.simulated == 'attribute':
        labeler = make_subreddit_based_simulated_labeler(FLAGS.beta0, FLAGS.beta1, FLAGS.gamma, FLAGS.simulation_mode,
                                                     seed=0)
    else:
        Exception("simulated flag not recognized")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    dev_splits = [int(s) for s in str.split(FLAGS.dev_splits)]
    test_splits = [int(s) for s in str.split(FLAGS.test_splits)]

    if FLAGS.subreddits == '':
        subreddits = None
    else:
        subreddits = [int(s) for s in FLAGS.subreddits.split(',')]

    train_input_fn = make_input_fn_from_file(
        input_files_or_glob=FLAGS.input_files,
        seq_length=FLAGS.max_seq_length,
        num_splits=FLAGS.num_splits,
        dev_splits=dev_splits,
        test_splits=test_splits,
        tokenizer=tokenizer,
        do_masking=do_masking,
        subreddits=subreddits,
        is_training=is_training,
        shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
        seed=FLAGS.seed,
        labeler=labeler,
        filter_train=is_training)

    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size

    dataset = train_input_fn(params={'batch_size': batch_size})

    # format expected by Keras for training
    if is_training:
        # dataset = filter_training(dataset)
        dataset = dataset.map(_keras_format, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(4)

    return dataset


def make_dragonnet_metrics():
    METRICS = [
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.AUC
    ]

    NAMES = ['binary_accuracy', 'precision', 'recall', 'auc']

    CONT_METRICS = [
        tf.keras.metrics.MeanSquaredError
    ]

    CONT_NAMES = ['mse']

    g_metrics = [m(name='metrics/' + n) for m, n in zip(METRICS, NAMES)]
    q0_metrics = [m(name='metrics/' + n) for m, n in zip(CONT_METRICS, CONT_NAMES)]
    q1_metrics = [m(name='metrics/' + n) for m, n in zip(CONT_METRICS, CONT_NAMES)]

    return {'g': g_metrics, 'q0': q0_metrics, 'q1': q1_metrics}


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.1')
    tf.random.set_seed(FLAGS.seed)

    # with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    #     input_meta_data = json.loads(reader.read().decode('utf-8'))

    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert20/'
    #
    # Configuration stuff
    #
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = FLAGS.num_train_epochs
    # train_data_size = 11778
    train_data_size = 5000
    steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)  # 368
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
    def _get_dragon_model(do_masking):
        if not FLAGS.fixed_feature_baseline:
            dragon_model, core_model = (
                bert_models.dragon_model(
                    bert_config,
                    max_seq_length=FLAGS.max_seq_length,
                    binary_outcome=False,
                    use_unsup=do_masking,
                    max_predictions_per_seq=20,
                    unsup_scale=1.))
        else:
            dragon_model, core_model = bert_models.derpy_dragon_baseline(
                bert_config,
                max_seq_length=FLAGS.max_seq_length,
                binary_outcome=False)

        # WARNING: the original optimizer causes a bug where loss increases after first epoch
        # dragon_model.optimizer = optimization.create_optimizer(
        #     FLAGS.train_batch_size * initial_lr, steps_per_epoch * epochs, warmup_steps)
        dragon_model.optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.train_batch_size * initial_lr)
        return dragon_model, core_model

    if (FLAGS.mode == 'train_and_predict') or (FLAGS.mode == 'train_only'): 
        # training. strategy.scope context allows use of multiple devices
        with strategy.scope():
            keras_train_data = make_dataset(is_training=True, do_masking=FLAGS.do_masking)

            dragon_model, core_model = _get_dragon_model(FLAGS.do_masking)
            optimizer = dragon_model.optimizer

            if FLAGS.init_checkpoint:
                checkpoint = tf.train.Checkpoint(model=core_model)
                checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()

            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
            if latest_checkpoint:
                dragon_model.load_weights(latest_checkpoint)

            dragon_model.compile(optimizer=optimizer,
                                 loss={'g': 'binary_crossentropy', 'q0': 'mean_squared_error',
                                       'q1': 'mean_squared_error'},
                                 loss_weights={'g': FLAGS.treatment_loss_weight, 'q0': 0.1, 'q1': 0.1},
                                 weighted_metrics=make_dragonnet_metrics())

            summary_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir, update_freq=128)
            checkpoint_dir = os.path.join(FLAGS.model_dir, 'model_checkpoint.{epoch:02d}')
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, period=10)

            callbacks = [summary_callback, checkpoint_callback]

            dragon_model.fit(
                x=keras_train_data,
                # validation_data=evaluation_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                # vailidation_steps=eval_steps,
                callbacks=callbacks)

        # save a final model checkpoint (so we can restore weights into model w/o training idiosyncracies)
        if FLAGS.model_export_path:
            model_export_path = FLAGS.model_export_path
        else:
            model_export_path = os.path.join(FLAGS.model_dir, 'trained/dragon.ckpt')

        checkpoint = tf.train.Checkpoint(model=dragon_model)
        saved_path = checkpoint.save(model_export_path)
    else:
        saved_path = FLAGS.saved_path

    # make predictions and write to file
    if FLAGS.mode != 'train_only':

        # create data and model w/o masking
        eval_data = make_dataset(is_training=False, do_masking=False)
        dragon_model, core_model = _get_dragon_model(do_masking=False)
        # reload the model weights (necessary because we've obliterated the masking)
        checkpoint = tf.train.Checkpoint(model=dragon_model)
        checkpoint.restore(saved_path).assert_existing_objects_matched()
        # loss added as simple hack to bizzarre keras bug that requires compile for predict, and a loss for compile
        dragon_model.add_loss(lambda: 0)
        dragon_model.compile()

        outputs = dragon_model.predict(x=eval_data)

        out_dict = {}
        out_dict['g'] = outputs[0].squeeze()
        out_dict['q0'] = outputs[1].squeeze()
        out_dict['q1'] = outputs[2].squeeze()

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
