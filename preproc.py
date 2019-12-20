import collections

import pandas as pd
pd.set_option('display.max_colwidth', -1)

import nlp.bert.tokenization as tokenization
import numpy as np
import tensorflow as tf


def reduce_to_context(conv):
    # conv = conv[['actor_id', 'id', 'interaction', 'message']].drop_duplicates()
    conv = conv[['interaction', 'message']].drop_duplicates()
    # print(conv[['interaction', 'message']])

    # include all text before the second message sent by counsellor
    # this is almost always the texter explaining the situation
    counselor_indices = conv[conv.interaction == 'counselor'].index
    if counselor_indices.size < 2:
        return None
    truncation_index = counselor_indices[1]

    texter = conv[conv.interaction == 'texter']
    texter_context = texter[texter.index < truncation_index]['message']
    # print(texter_context[['message']])

    # concat strings, separated by BERT special tokens
    # texter_context = texter_context.str.cat(sep=' [SEP] ')
    # texter_context = '[CLS] ' + texter_context + ' [SEP]'

    return texter_context


def bert_proc_context(context, tokenizer, max_seq_length):
    def _tokenize_sentence(sentence):
        sanitized = tokenization.convert_to_unicode(sentence)
        sanitized = sanitized.strip()
        tokens = tokenizer.tokenize(sanitized)
        tokens.append("[SEP]")

        return tokens
    tokens = context.transform(_tokenize_sentence)
    tokens = tokens.agg('sum')
    tokens = ["[CLS]"] + tokens

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]

    input_mask = [1] * len(input_ids)
    segment_ids = [0] * max_seq_length

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto,
    e.g, An integer label.
    """
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tf_example(input_ids, input_mask, segment_ids, conversation_id):
    """
    Parses the input paper into a tf.Example as expected by Bert
    Note: the docs for tensorflow Example are awful ¯\_(ツ)_/¯
    """

    features = collections.OrderedDict()
    features["input_ids"] = _int64_feature(input_ids)
    features["input_mask"] = _int64_feature(input_mask)
    features["segment_ids"] = _int64_feature(segment_ids)
    features["conversation_id"] = _int64_feature(conversation_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    return tf_example


def main():

    with pd.HDFStore('local_dat.h5') as store:
        df = store['small_samp']

    conversations = df.groupby('conversation_id')
    contexts = conversations.apply(reduce_to_context)
    contexts = contexts.dropna()

    vocab_file = 'pretrained_models/keras_bert/cased_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=False)

    def _bert_proc_and_serialize(context, conversation_id):
        context_dict = bert_proc_context(context, tokenizer, 50)
        tf_example = make_tf_example(**context_dict, conversation_id=conversation_id)
        return tf_example.SerializeToString()
    processed = contexts.groupby('conversation_id').apply(lambda x: _bert_proc_and_serialize(x, x.name))

    with tf.io.TFRecordWriter('test_file.tf_record') as writer:
        for serialized in list(processed):
            writer.write(serialized)


if __name__ == '__main__':
    main()