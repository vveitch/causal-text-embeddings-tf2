import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

import tensorflow as tf

from semi_parametric_estimation.att import att_estimates

pd.set_option('display.max_colwidth', -1)


def att_from_bert_tsv(tsv_path, test_split=True):
    predictions = pd.read_csv(tsv_path, sep='\t')

    if test_split:
        reduced_df = predictions[predictions.in_test == 1]
    else:
        reduced_df = predictions[predictions.in_train == 1]


    gt = reduced_df[reduced_df.treatment == 1].y1.mean() - reduced_df[reduced_df.treatment == 1].y0.mean()
    # print(f"Ground truth: {gt}")

    naive = reduced_df[reduced_df.treatment == 1].outcome.mean() - reduced_df[reduced_df.treatment == 0].outcome.mean()
    # print(f"Naive: {naive}")

    selections = {'y': 'outcome',
                  't': 'treatment',
                  'q0': 'q0',
                  'q1': 'q1',
                  'g': 'g'}

    reduced_df = reduced_df[selections.values()]
    rename_dict = {v: k for k, v in selections.items()}
    reduced_df = reduced_df.rename(columns=rename_dict)

    # get rid of any sample w/ less than 1% chance of receiving treatment
    # include_sample = reduced_df['g'] > 0.01
    # reduced_df = reduced_df[include_sample]

    nuisance_dict = reduced_df.to_dict('series')
    nuisance_dict['prob_t'] = nuisance_dict['t'].mean()
    estimates = att_estimates(**nuisance_dict, deps=0.00005)

    estimates['ground_truth'] = gt
    # estimates['naive'] = naive

    return estimates


def dragon_att(output_dir, test_split=True):
    """
    Expects that the data was split into k folds, and the predictions from each fold
    was saved in experiment_dir/[fold_identifier]/[prediction_file].tsv.

    :param output_dir:
    :return:
    """

    data_files = sorted(glob.glob(f'{output_dir}/*/*.tsv', recursive=True))
    estimates = []
    for data_file in data_files:
        try:
            all_estimates = att_from_bert_tsv(data_file, test_split=test_split)
            # print(psi_estimates)
            estimates += [all_estimates]
        except:
            print('wtf')
            print(data_file)

    avg_estimates = {}
    for k in all_estimates.keys():
        k_estimates = []
        for estimate in estimates:
            k_estimates += [estimate[k]]

        if "trim_test":
            k_estimates = np.sort(k_estimates)[1:-1]
        avg_estimates[k] = np.mean(k_estimates)
        avg_estimates[(k, 'std')] = np.std(k_estimates)
        # w/ test split, we want standard deviation of the mean (our estimate)
        # w/o test split, each value is a valid estimate, so we just want entry-wise std
        if test_split:
            avg_estimates[(k, 'std')] /= np.sqrt(len(k_estimates))

    return avg_estimates


def confounding_level():
    # Comparison over compounding strength
    estimates = {}
    estimates['low'] = dragon_att('../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta11.0.gamma0.0')
    estimates['med'] = dragon_att('../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta15.0.gamma0.0')
    estimates['high'] = dragon_att('../out/PeerRead/buzzy-based-sim/modesimple/beta00.25.beta125.0.gamma0.0')

    estimate_df = pd.DataFrame(estimates)
    with tf.io.gfile.GFile('../out/PeerRead/buzzy-based-sim/estimates.tsv', "w") as writer:
        writer.write(estimate_df.to_csv(sep="\t"))

    print(estimate_df.round(2))


def buzzy_baselines():
    base_dir = '../out/PeerRead/buzzy-baselines/'
    out_file = 'modesimple/beta00.25.beta15.0.gamma0.0'

    estimates = {}
    estimates['baseline'] = dragon_att(os.path.join(base_dir, 'buzzy', out_file))
    estimates['fixed_features'] = dragon_att(os.path.join(base_dir, 'fixed-features', out_file))
    estimates['no_pretrain'] = dragon_att(os.path.join(base_dir, 'no-init', out_file))
    estimates['no_masking'] = dragon_att(os.path.join(base_dir, 'no-masking', out_file))
    estimates['no_dragon'] = dragon_att(os.path.join(base_dir, 'no-dragon', out_file))


    estimate_df = pd.DataFrame(estimates)
    print(estimate_df.round(2))


def real():
    estimates={}
    estimates['buzzy'] = dragon_att('../out/PeerRead/real/o_accepted_t_buzzy_title')
    estimates['theorem_referenced'] = dragon_att('../out/PeerRead/real/o_accepted_t_theorem_referenced')
    estimate_df = pd.DataFrame(estimates)
    print(estimate_df.round(2))

def main():
    # estimates = confounding_level()
    # estimates = real()
    estimates = buzzy_baselines()

if __name__ == '__main__':
    main()
    # pass
