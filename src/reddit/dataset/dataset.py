"""
TODO:

Main trick is getting the sentence matching task to work.
Probably the easiest thing is:
 1. write dataset with the 'natural' (i.e., tf_record) structure
 2. adapt this by drawing a batch of size 2, and randomly swapping examples.
    (but I'm not 100% sure how to do this)
 3. ?????
 4. profit.

"""
import argparse
import pandas as pd
import tensorflow as tf

try:
    import mkl_random as random
except ImportError:
    import numpy.random as random

from tf_official.nlp.bert import tokenization
from reddit.dataset.sentence_masking import create_masked_lm_predictions
import numpy as np
from scipy.special import logit, expit

# hardcoded because protobuff is not self describing for some bizarre reason
# all features are (lists of) tf.int64
# all_context_features = ['op_id',
#                         'op_gender',
#                         'post_id',
#                         'subreddit',
#                         'op_gender_visible',
#                         'responder_id',
#                         'has_random_resp'
#                         # 'responder_gender',
#                         # 'responder_gender_visible'
#                         ]

num_distinct_subreddits = 20

all_context_features = ['author',
                        'author_flair_css_class',
                        'author_flair_text',
                        'controversiality',
                        'created_utc',
                        'gender',
                        'gilded',
                        'id',
                        'link_id',
                        'parent_id',
                        'score',
                        'subreddit',
                        'has_random_resp',
                        'many_split',
                        'index']


def get_one_hot_encoding(subreddit_idx):
    return tf.one_hot(subreddit_idx, num_distinct_subreddits)


def compose(*fns):
    """ Composes the given functions in reverse order.

    Parameters
    ----------
    fns: the functions to compose

    Returns
    -------
    comp: a function that represents the composition of the given functions.
    """
    import functools

    def _apply(x, f):
        if isinstance(x, tuple):
            return f(*x)
        else:
            return f(x)

    def comp(*args):
        return functools.reduce(_apply, fns, args)

    return comp


'''
DS: adding a number of data processing fns for supervised task
'''

def make_null_labeler():
    """
    labeler function that returns meaningless labels. Convenient for pre-training, where the labels are totally unused
        :return:
    """

    def labeler(data):
        return {**data, 'outcome': tf.zeros([1]), 'y0': tf.zeros([1]), 'y1': tf.zeros([1]), 'treatment': tf.zeros([1])}

    return labeler


def make_real_labeler(treatment_name, outcome_name):
    def labeler(data):
        return {**data, 'outcome': data[outcome_name], 'treatment': data[treatment_name], 'y0': tf.zeros([1]),
                'y1': tf.zeros([1])}

    return labeler

def outcome_sim(beta0, beta1, gamma, treatment, confounding, noise, setting="simple"):
    if setting == "simple":
        y0 = beta1 * confounding
        y1 = beta0 + y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "multiplicative":
        y0 = beta1 * confounding
        y1 = beta0 * y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "interaction":
        # required to distinguish ATT and ATE
        y0 = beta1 * confounding
        y1 = y0 + beta0 * tf.math.square(confounding + 0.5)

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    else:
        raise NotImplemented('setting argument to make_simulated_labeler not recognized')

    return simulated_score, y0, y1


def make_subreddit_standardized_scores():
    # standardize the scores on a per-subreddit basis to deal w/ crazy convergence issues

    # hardcoded because life is suffering
    all_mean_scores = np.array([4.16580311, 61.08446939, 19.64020508, 43.51536639, 16.83549653,
                                11.37175703, 7.91225941, 54.5513382, 41.62996557, 61.8696114,
                                21.74156836, 4.26996656, 6.3032812, 6.33557766, 10.75942321,
                                12.29322831, 7.86248355, 6.96632453, 14.13948171, 6.62278865], dtype=np.float32)
    all_std_scores = np.array([4.25777127, 358.8567717,  62.02295383, 203.28024083,
                               26.27968206,  21.55731166,  16.68331688, 329.31646208,
                               105.19197613, 115.06941069,  70.2522788,   5.20143709,
                               9.37623701,  30.44100267,  69.05652112,  20.15282915,
                               14.01754684,  11.22911321,  33.31924185,  11.04199622], dtype=np.float32)

    def labeler(data):
        subreddit_idx = data['subreddit']
        score = tf.cast(data['score'], tf.float32)

        mean_score = tf.gather(all_mean_scores, subreddit_idx)
        std_score = tf.gather(all_std_scores, subreddit_idx)

        standardized = (score - mean_score) / std_score

        return {**data, 'standard_score': standardized}

    return labeler


def make_log_scores():
    # standardize the scores on a per-subreddit basis to deal w/ crazy convergence issues

    def labeler(data):
        score = tf.cast(data['score'], tf.float32)

        log_score = tf.log(tf.nn.relu(score)+1.)
        return {**data, 'log_score': log_score}

    return labeler


def make_subreddit_based_simulated_labeler(treat_strength, con_strength, noise_level, setting="simple", seed=42):
    # hardcode gender proportions of each subreddit :'(
    gender_props = np.array(
        [0.08290155440414508, 0.9306885544915641, 0.9444306623666584, 0.053265121877821245, 0.0836100211288862,
         0.9018952928382787, 0.6491243280735217, 0.7985401459854015, 0.3436175847457627, 0.2293529255554572,
         0.7604441360166551, 0.04929765886287625, 0.6117755289788408, 0.515695067264574, 0.24193122130091507,
         0.06660675582809114, 0.5266344888108819, 0.875792872794372, 0.8210111788617886, 0.0022985674998973853], dtype=np.float32)

    np.random.seed(seed)
    all_noise = np.array(random.normal(0, 1, 422206), dtype=np.float32)

    def labeler(data):

        subreddit_idx = data['subreddit']
        index = data['index']
        treatment = data['gender']
        treatment = tf.cast(treatment, tf.float32)
        confounding = tf.gather(gender_props, subreddit_idx) - 0.5
        noise = tf.gather(all_noise, index)

        simulated_score, y0, y1 = outcome_sim(treat_strength, con_strength, noise_level, treatment, confounding, noise,
                                              setting=setting)

        return {**data, 'outcome': simulated_score,  'treatment': treatment, 'y0': y0, 'y1': y1} 
        #, 'confounding': confounding}

    return labeler


def make_propensity_based_simulated_labeler(treat_strength, con_strength, noise_level,
                                            base_propensity_scores, example_indices, exogeneous_con=0.,
                                            setting="simple", seed=42):
    np.random.seed(seed)
    all_noise = random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)
    # extra_confounding = random.binomial(1, 0.5*np.ones_like(base_propensity_scores)).astype(np.float32)
    extra_confounding = random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)

    all_propensity_scores = expit((1.-exogeneous_con)*logit(base_propensity_scores) + exogeneous_con * extra_confounding).astype(np.float32)
    all_treatments = random.binomial(1, all_propensity_scores).astype(np.int32)

    # indices in dataset refer to locations in entire corpus,
    # but propensity scores will typically only inlcude a subset of the examples
    reindex_hack = np.zeros(422206, dtype=np.int32)
    reindex_hack[example_indices] = np.arange(example_indices.shape[0], dtype=np.int32)

    def labeler(data):
        index = data['index']
        index_hack = tf.gather(reindex_hack, index)
        treatment = tf.gather(all_treatments, index_hack)
        confounding = tf.gather(all_propensity_scores, index_hack) - 0.5
        noise = tf.gather(all_noise, index_hack)

        simulated_score, y0, y1 = outcome_sim(treat_strength, con_strength, noise_level, tf.cast(treatment, tf.float32),
                                              confounding, noise, setting=setting)

        return {**data, 'outcome': simulated_score, 'y0': y0, 'y1': y1, 'treatment': treatment}

    return labeler


def make_split_document_labels(num_splits, dev_splits, test_splits):
    """
    Adapts tensorflow dataset_ to produce additional elements that indicate whether each datapoint is in train, dev,
    or test

    Particularly, splits the data into num_split folds, and censors the censored_split fold

    Parameters
    ----------
    num_splits integer in [0,100)
    dev_splits list of integers in [0,num_splits)
    test_splits list of integers in [0, num_splits)

    Returns
    -------
    fn: A function that can be used to map a dataset_ to censor some of the document labels.
    """

    def _tf_in1d(a, b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        a = tf.expand_dims(a, 0)
        b = tf.expand_dims(b, 1)
        return tf.reduce_any(input_tensor=tf.equal(a, b), axis=1)

    def _tf_scalar_a_in1d_b(a, b):
        """
        Tensorflow equivalent of np.in1d(a,b)
        """
        return tf.reduce_any(input_tensor=tf.equal(a, b))

    def fn(data):
        many_split = data['many_split']
        reduced_split = tf.math.floormod(many_split, num_splits)  # reduce the many splits to just num_splits

        in_dev = _tf_scalar_a_in1d_b(reduced_split, dev_splits)
        in_test = _tf_scalar_a_in1d_b(reduced_split, test_splits)
        in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # in_dev = _tf_in1d(reduced_splits, dev_splits)
        # in_test = _tf_in1d(reduced_splits, test_splits)
        # in_train = tf.logical_not(tf.logical_or(in_dev, in_test))

        # code expects floats
        in_dev = tf.cast(in_dev, tf.float32)
        in_test = tf.cast(in_test, tf.float32)
        in_train = tf.cast(in_train, tf.float32)

        return {**data, 'in_dev': in_dev, 'in_test': in_test, 'in_train': in_train}

    return fn


def _make_bert_compatifier(do_masking):
    """
    Makes a parser to change feature naming to be consistent w/ what's expected by pretrained bert model, i.e. filter
    down to only features used as bert input, and put data into expected (features, labels) format
    """

    def bert_compatibility(data):
        # data['input_word_ids'] = data.pop('maybe_masked_input_ids')
        # data['input_mask'] = data.pop('token_mask')

        if do_masking:
            x = {
                'input_word_ids': data['maybe_masked_input_ids'],
                'input_mask': data['op_token_mask'],
                'input_type_ids': tf.zeros_like(data['op_token_mask']),  # segment ids
                'masked_lm_positions': data['masked_lm_positions'],
                'masked_lm_ids': data['masked_lm_ids'],
                'masked_lm_weights': data['masked_lm_weights'],
                # next_sentence_label = 1 if instance.is_random_next else 0
                'next_sentence_labels': tf.constant([0], tf.int32)
            }

            # y = data['masked_lm_weights']

        else:
            x = {
                'input_word_ids': data['maybe_masked_input_ids'],
                'input_mask': data['op_token_mask'],
                'input_type_ids': tf.zeros_like(data['op_token_mask']),  # segment ids
            }

        y = {'outcome': data['outcome'], 'treatment': data['treatment'],
             'in_dev': data['in_dev'], 'in_test': data['in_test'], 'in_train': data['in_train'],
             'y0': data['y0'], 'y1': data['y1'],
             'index': data['index']}

        return x, y

    return bert_compatibility


def dataset_processing(dataset, parser, masker, labeler, do_masking, is_training, num_splits, dev_splits, test_splits, batch_size,
                       filter_test=False,
                       subreddits=None,
                       shuffle_buffer_size=100):
    """

    Parameters
    ----------
    dataset  tf.data dataset
    parser function, read the examples, should be based on tf.parse_single_example
    masker function, should provide Bert style masking
    labeler function, produces labels
    is_training
    num_splits
    censored_split
    batch_size
    filter_test restricts to only examples where in_test=1
    shuffle_buffer_size

    Returns
    -------

    """

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    data_processing = compose(parser,  # parse from tf_record
                              labeler,  # add a label (unused downstream at time of comment)
                              make_split_document_labels(num_splits, dev_splits, test_splits),  # censor some labels
                              masker,
                              _make_bert_compatifier(do_masking))  # Bert style token masking for unsupervised training

    dataset = dataset.map(data_processing, 4)

    if subreddits is not None:
        def filter_fn(data):
            filter = False
            for subreddit in subreddits:
                filter = tf.logical_or(filter, tf.equal(data['subreddit'], subreddit))

            return filter

        dataset = dataset.filter(filter_fn)

    if filter_test:
        def filter_fn(data):
            return tf.equal(data['in_test'], 1)

        dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset


def null_masker(data):
    return {
        **data,
        'maybe_masked_input_ids': data['token_ids'],
        'masked_lm_positions': tf.zeros_like(data['token_ids']),
        'masked_lm_ids': tf.zeros_like(data['token_ids']),
        'masked_lm_weights': tf.zeros_like(data['token_ids'])
    }

def make_input_fn_from_file(input_files_or_glob, seq_length,
                            num_splits, dev_splits, test_splits,
                            tokenizer, is_training,
                            do_masking=True,
                            filter_test=False,
                            subreddits=None,
                            shuffle_buffer_size=int(1e6), seed=0, labeler=None):
    input_files = []
    for input_pattern in input_files_or_glob.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    if labeler is None:
        labeler = make_null_labeler()

    if do_masking:
        masker = make_input_id_masker(tokenizer, seed)  # produce masked subsets for unsupervised training
    else:
        masker = null_masker

    def input_fn(params):
        batch_size = params["batch_size"]

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=len(input_files))
            cycle_length = min(4, len(input_files))

        else:
            dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            cycle_length = 1  # go through the datasets in a deterministic order

        # make the record parsing ops
        max_abstract_len = seq_length

        parser = make_parser(abs_seq_len=max_abstract_len)  # parse the tf_record
        encoder = make_one_hot_encoder()

        parser = compose(parser, encoder)

        # for use with interleave
        def _dataset_processing(input):
            input_dataset = tf.data.TFRecordDataset(input)
            processed_dataset = dataset_processing(input_dataset,
                                                   parser, masker, labeler,
                                                   do_masking,
                                                   is_training,
                                                   num_splits, dev_splits, test_splits,
                                                   batch_size,
                                                   filter_test=filter_test,
                                                   subreddits=subreddits,
                                                   shuffle_buffer_size=shuffle_buffer_size)
            return processed_dataset

        dataset = dataset.interleave(_dataset_processing)
        return dataset

    return input_fn


def make_input_id_masker(tokenizer, seed):
    # (One of) Bert's unsupervised objectives is to mask some fraction of the input words and predict the masked words

    def masker(data):
        token_ids = data['op_token_ids']
        maybe_masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = create_masked_lm_predictions(
            token_ids,
            # pre-training defaults from Bert docs
            masked_lm_prob=0.15,
            max_predictions_per_seq=20,
            vocab=tokenizer.vocab,
            seed=seed)
        return {
            **data,
            'maybe_masked_input_ids': maybe_masked_input_ids,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_lm_weights': masked_lm_weights
        }
    return masker


def make_one_hot_encoder():
    # (One of) Bert's unsupervised objectives is to mask some fraction of the input words and predict the masked words

    def encoder(data):
        subreddit_idx = data['subreddit']
        subreddit_encoding = get_one_hot_encoding(subreddit_idx)
        return {
            **data,
            'subreddit_encoding': subreddit_encoding
        }

    return encoder


def make_parser(abs_seq_len=128):
    context_features = {cf: tf.io.FixedLenFeature([], dtype=tf.int64) for cf in all_context_features}

    # TODO: check that our segment_ids convention matches Bert
    text_features = {
        "op_token_ids": tf.io.FixedLenFeature([abs_seq_len], tf.int64),
        "op_token_mask": tf.io.FixedLenFeature([abs_seq_len], tf.int64)
        # "segment_ids": tf.io.FixedLenFeature([abs_seq_len], tf.int64)
    }

    _name_to_features = {**context_features, **text_features}

    def parser(record):
        tf_example = tf.io.parse_single_example(
            record,
            features=_name_to_features
        )

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(tf_example.keys()):
            t = tf_example[name]

            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            tf_example[name] = t

        return tf_example

    return parser


def make_normalize_score():
    """
    Hardcoded hack to deal w/ variability of score
    :return:
    """

    def normalize_score(data):
        score = data['score']
        norm_score = (score - 24.3) / 166.7

        return {**data,
                'normalized_score': norm_score}

    return normalize_score


def make_input_fn_from_tfrecord(tokenizer, tfrecord='../dat/reddit/proc.tf_record',
                                is_training=True, shuffle_buffer_size=1e6,
                                seed=0):
    def input_fn(params):
        batch_size = params["batch_size"]
        dataset = tf.data.TFRecordDataset(tfrecord)

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # functions for additional processing
        basic_parser = make_parser()  # parse from tf_record
        unsupervised_parser = make_bert_unsupervised_parser()  # concats op and response together
        masker = make_input_id_masker(tokenizer, seed)
        encoder = make_one_hot_encoder()

        data_processing = compose(basic_parser,
                                  unsupervised_parser,
                                  masker,
                                  encoder)

        dataset = dataset.map(data_processing, 4)

        # def filter_test_fn(data):
        #     return tf.equal(data['subreddit'], 1)

        # dataset = dataset.filter(filter_test_fn)

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        return dataset

    return input_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_buffer_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_abs_len', type=int, default=128)

    args = parser.parse_args()

    # for easy debugging
    # tsv_file = "../../dat/PeerRead/proc/acl_2017.tf_record"
    # tsv_file = glob.glob('/home/victor/Documents/causal-spe-embeddings/dat/PeerRead/proc/*.tf_record')
    filename = '/proj/sml_netapp/dat/undocumented/reddit/proc.tf_record'

    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    #                             trainable=True)
    # vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    vocab_file = '/proj/sml_netapp/projects/victor/causal-text-tf2/pre-trained/uncased_L-12_H-768_A-12/vocab.txt'

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    num_splits = 10
    # dev_splits = [0]
    # test_splits = [0]
    dev_splits = []
    test_splits = [1, 2]

    input_dataset_from_filenames = make_input_fn_from_file(filename,
                                                             args.max_abs_len,
                                                             num_splits,
                                                             dev_splits,
                                                             test_splits,
                                                             tokenizer,
                                                             do_masking=True,
                                                             is_training=True,
                                                             filter_test=False,
                                                             shuffle_buffer_size=25000,
                                                             labeler=None,
                                                             seed=0)
    params = {'batch_size': 10000}
    dataset = input_dataset_from_filenames(params)

    print(dataset.element_spec)

    for val in dataset.take(1):
        sample = val

    sample = next(iter(dataset))
    print(sample)

    # print(sample[0]['masked_lm_ids'])


if __name__ == "__main__":
    main()
