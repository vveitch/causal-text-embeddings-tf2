from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from tf_official.nlp.bert import tokenization

MININT = -2147483648


# Masking
def _make_input_id_masker(tokenizer, seed,
                          masked_lm_prob=0.15,
                          max_predictions_per_seq=20):
    """

    :param tokenizer: tokenizer used for pre-processing (required to avoid masking special tokens)
    :param seed: random seed
    :param masked_lm_prob: per-token probability of masking
    :param max_predictions_per_seq: maximum allowed number of masks
    :return:
    """
    # (One of) Bert's unsupervised objectives is to mask some fraction of the input words and predict the masked words
    vocab = tokenizer.vocab
    @tf.function
    def _create_masked_lm_predictions(token_ids: tf.Tensor):
        """
        Randomly masks tokens to produce (one of) the BERT unsupervised learning objectives

        Args:
            token_ids: the ids to be masked

        Returns: output_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights as expected by BERT model

        """

        basic_mask = tf.less(
            tf.random.uniform(token_ids.shape, minval=0, maxval=1, dtype=tf.float32, seed=seed),
            masked_lm_prob)

        # don't mask special characters or padding
        cand_indexes = tf.logical_and(tf.not_equal(token_ids, vocab["[CLS]"]),
                                      tf.not_equal(token_ids, vocab["[SEP]"]))
        cand_indexes = tf.logical_and(cand_indexes, tf.not_equal(token_ids, 0))
        mask = tf.logical_and(cand_indexes, basic_mask)

        # sometimes nothing gets masked. In that case, just mask the first valid token
        masked_lm_positions = tf.cond(pred=tf.reduce_any(mask),
                                      true_fn=lambda: tf.where(mask),
                                      false_fn=lambda: tf.where(cand_indexes)[0:2])

        masked_lm_positions = masked_lm_positions[:, 0]

        # truncate to max predictions for ease of padding
        masked_lm_positions = tf.random.shuffle(masked_lm_positions, seed=seed)
        masked_lm_positions = masked_lm_positions[0:max_predictions_per_seq]
        masked_lm_positions = tf.cast(masked_lm_positions, dtype=tf.int32)
        masked_lm_ids = tf.gather(token_ids, masked_lm_positions)

        mask = tf.cast(
            tf.scatter_nd(tf.expand_dims(masked_lm_positions, 1), tf.ones_like(masked_lm_positions), token_ids.shape),
            bool)

        output_ids = tf.where(mask, vocab["[MASK]"] * tf.ones_like(token_ids), token_ids)

        # pad out to max_predictions_per_seq
        masked_lm_weights = tf.ones_like(masked_lm_ids, dtype=tf.float32)  # tracks padding
        add_pad = [[0, max_predictions_per_seq - tf.shape(input=masked_lm_positions)[0]]]
        masked_lm_weights = tf.pad(tensor=masked_lm_weights, paddings=add_pad, mode='constant')
        masked_lm_positions = tf.pad(tensor=masked_lm_positions, paddings=add_pad, mode='constant')
        masked_lm_ids = tf.pad(tensor=masked_lm_ids, paddings=add_pad, mode='constant')

        return output_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    def masker(data, label=None):
        token_ids = data['input_word_ids']
        maybe_masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = _create_masked_lm_predictions(
            token_ids)
        data['input_word_ids'] = maybe_masked_input_ids
        data['masked_lm_positions'] = masked_lm_positions
        data['masked_lm_ids'] = masked_lm_ids
        data['masked_lm_weights'] = masked_lm_weights

        # official BERT models require next sentence label inputs, so we spoof some nonsense
        if not 'next_sentence_labels' in data.keys():
            batch_size = 1 if data['input_word_ids']._rank() == 1 else data['input_word_ids'].shape[0]
            data['next_sentence_labels'] = tf.zeros(batch_size)

        return data, label if label else data

    return masker


def add_masking(dataset: tf.data.Dataset,
                tokenizer: tokenization.FullTokenizer,
                masked_lm_prob=0.15,
                max_predictions_per_seq=20,
                seed=0):
    """
    Applies Bert style word-piece masking to input dataset, using the provided tokenizer and random seed
    Args:
        dataset: dataset, should yield dictionary data that has key 'input_ids'
        tokenizer: bert tokenizer used for pre-processing
        masked_lm_prob: per-token masking probability
        max_predictions_per_seq: maximum allowed number of masks
        seed: random seed

    Returns: tensorflow dataset
    """
    masker = _make_input_id_masker(tokenizer, seed,
                                   masked_lm_prob=masked_lm_prob, max_predictions_per_seq=max_predictions_per_seq)
    mask_dataset = dataset.map(
        masker, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return mask_dataset


# basic data loading
def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def load_basic_bert_data(input_files_or_globs,
                         seq_length,
                         is_training=True,
                         input_pipeline_context=None):
    name_to_features = {
        'token_ids':
            tf.io.FixedLenFeature([seq_length], tf.int64),
        'token_mask':
            tf.io.FixedLenFeature([seq_length], tf.int64),
        'id':
            tf.io.FixedLenFeature([1], tf.int64),
    }

    input_files = []
    for input_pattern in input_files_or_globs.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    dataset = tf.data.Dataset.from_tensor_slices(input_files)

    # dataset = tf.data.Dataset.list_files(input_files, shuffle=is_training)

    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                                input_pipeline_context.input_pipeline_id)

    if is_training:
        dataset = dataset.repeat()
        # We set shuffle buffer to exactly match total number of
        # training files to ensure that training data is well shuffled.
        dataset = dataset.shuffle(len(input_files))

    # In parallel, create tf record dataset for each train files.
    # cycle_length = 8 means that up to 8 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    cycle_length = min(len(input_files), 8)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    decode_fn = lambda record: _decode_record(record, name_to_features)
    dataset = dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _bert_conventions(data):
        data['input_word_ids'] = data.pop('token_ids')
        data['input_mask'] = data.pop('token_mask')
        data['input_type_ids'] = tf.zeros_like(data['input_mask'])  # fake segment ids
        return data

    dataset = dataset.map(_bert_conventions, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def dataset_to_pandas_df(dataset):

    samp_dict = defaultdict(list)

    for sample in iter(dataset):
        for k, v in sample.items():
            samp_dict[k] += [v.numpy().squeeze()]

    proto_dict = {}
    for k, v in samp_dict.items():
        proto_dict[k] = np.concatenate(v)

    df = pd.DataFrame(proto_dict)

    return df


# moderately faster sampling
def _make_labeling_v2(label_df: pd.DataFrame, do_factorize=False):
    """
    helper function for dataset_labels_from_pandas
    """
    label_df = label_df.copy()
    label_df = label_df.sort_values('id')  # so we can use tf.search_sorted

    ids = tf.convert_to_tensor(label_df.id, tf.int32)
    label_df = label_df.drop(columns=['id'])

    if do_factorize:
        def _factorize_nonfloats(col):
            if not np.issubdtype(col, np.floating):
                return pd.factorize(col)[0]
            else:
                return col

        label_df = label_df.apply(_factorize_nonfloats, axis=0)

    int_df = label_df.select_dtypes(include='int')
    int_names = list(int_df.keys())
    int_values = tf.convert_to_tensor(int_df.values, dtype=tf.int32)
    num_ints = len(int_names)

    float_df = label_df.select_dtypes(include='float')
    float_names = list(float_df.keys())
    float_values = tf.convert_to_tensor(float_df.values, dtype=tf.float32)
    num_floats = len(float_names)

    @tf.function
    def get_labels(id: tf.Tensor):
        idx = tf.searchsorted(ids, tf.transpose(id), out_type=tf.int32)  # id should be [batch_size, 1], so must transp
        idx = tf.transpose(idx)  # to get to shape [batch_size, 1]

        in_label_df = tf.equal(id, tf.gather(ids, idx))

        labels = {}

        if num_ints > 0:
            samp_int_vals = tf.gather(int_values, idx)
            samp_int_list = tf.unstack(samp_int_vals, axis=-1)
            for n, v in zip(int_names, samp_int_list):
                labels[n] = v

        if num_floats > 0:
            samp_float_vals = tf.gather(float_values, idx)
            samp_float_list = tf.unstack(samp_float_vals, axis=-1)
            for n, v in zip(float_names, samp_float_list):
                labels[n] = v

        labels['in_label_df'] = in_label_df
        labels['id'] = id

        return labels

    def labeling(data: dict, labels=None):
        id = data['id']
        added_labels = get_labels(id)
        labels = {**labels, **added_labels} if labels else added_labels
        return data, labels

    return labeling


def dataset_labels_from_pandas(dataset: tf.data.Dataset, label_df: pd.DataFrame, filter_labeled=True, do_factorize=False):
    """
    Produce a tensorflow dataset with labels by merging a dataset without labels and pandas dataframe

    takes a tensorflow dataset that yields a dictionary (i.e., features only) where one key of that dictionary is 'id',
    and a pandas dataframe where one column is 'id', and the other columns are labels of the example w/ that id.
    Returns a tensorflow dataset where examples include the labels.
    Typical use case is when a model has complicated unsupervised pre-processing which may be used for many possible
    prediction tasks

    Args:
        dataset: tf.data.Dataset. Must yield a dictionary which has key 'index'
        label_df: pd.DataFrame. The index should contain (a subset of) the values of 'index' in the tensorflow dataset.
            Each column should either be categorical or numerical.
        filter_labeled: bool. If True, then any batch that has 1 or more examples missing labels is filtered
            WARNING: this may result in very slow performance for large batches if missing labels are common.
            consider applying batching after this function
        cache: string. if present, dataset with labels will be cached to this file. If 'True', will be cached to memory
        do_factorize: bool. If True, then all non-float columns will be factorized (NaNs get mapped to -1)
            It's probably best practice to do this manually and not use the automation

    Returns: tf.data.Dataset that yields (original, labels_from_pandas)

    """
    labeling = _make_labeling_v2(label_df, do_factorize=do_factorize)
    lab_dataset = dataset.map(
        labeling, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if filter_labeled:
        lab_dataset = lab_dataset.filter(lambda d, l: tf.reduce_all(l['in_label_df']))

    return lab_dataset


def main():
    # label_df = pd.read_feather('dat/PeerRead/proc/arxiv-all-labels-only.feather')

    dataset = load_basic_bert_data('dat/PeerRead/proc/arxiv-all.tf_record', 250,
                                   is_training=True)

    label_df = pd.read_feather('dat/PeerRead/proc/arxiv-all-multi-treat-and-missing-outcomes.feather')
    dataset = dataset_labels_from_pandas(dataset, label_df)
    #
    dataset = dataset.cache()

    tokenizer = tokenization.FullTokenizer(
        vocab_file='pre-trained/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    dataset = add_masking(dataset, tokenizer=tokenizer)

    dataset.batch(32)
    dataset.shuffle(100)

    dit = dataset.take(100000)
    s = next(iter(dit))
    t0 = time.time()
    for samp in dit:
        s = samp
    t1 = time.time()
    print(t1 - t0)

    # data = make_unprocessed_PeerRead_dataset('../dat/PeerRead/proc/arxiv-all.tf_record', 250)
    # label_df = dataset_to_pandas_df(data)
    # label_df = label_df.drop(columns=['token_ids', 'token_mask', 'index'])
    # label_df.to_feather('../dat/PeerRead/proc/arxiv-all-labels-only.feather')


if __name__ == "__main__":
    main()
