import pandas as pd
import numpy as np
import tensorflow as tf
from semi_parametric_estimation.att import tmle_missing_outcomes

from .data_utils import dataset_to_pandas_df


def add_splits_to_label_df(label_df_path):
    label_df = pd.read_feather(label_df_path)
    splits = np.random.randint(0, 100, size=label_df.shape[0])
    label_df['many_split'] = splits
    label_df.to_feather(label_df_path)


def hydra_predict_and_write_to_csv(all_eval_data: tf.data.Dataset,
                                   hydra_model: tf.keras.Model,
                                   missing_outcomes: bool,
                                   prediction_file: str,
                                   num_shards=100):
    """
    Make predictions from eval_data using hydra_model, and write them to a tsv at prediction file

    :param all_eval_data: data to make predictions for
    :param hydra_model: model to use for predictions
    :param missing_outcomes: whether there are missing outcomes (changes signature of data labels and model output)
    :param prediction_file: file to write tsv
    :param num_shards: running predictions on large data can take a long time, so we shard the data into num_shards
        as a form of checkpointing. Each shard will have an intermediate TSV written out
    :return:
    """
    # running evaluation on very large data can take an obscenely long time, so we shard the data as checkpointing
    for idx in range(num_shards):
        eval_data = all_eval_data.shard(num_shards=num_shards, index=idx)

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

        label_dataset = eval_data.map(lambda f, l: l, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data_df = dataset_to_pandas_df(label_dataset)

        outs = data_df.join(predictions)
        with tf.io.gfile.GFile(prediction_file + f'_index{idx}', "w") as writer:
            writer.write(outs.to_csv(sep="\t"))

    # merge shards and write final predictions csv
    dfs = []
    for idx in range(num_shards):
        dfs.append(pd.read_csv(prediction_file + f'_index{idx}', sep='\t'))

    full_df = pd.concat(dfs)
    with tf.io.gfile.GFile(prediction_file, "w") as writer:
        writer.write(full_df.to_csv(sep="\t"))


def hydra_att_from_predictions_missing_outcomes(prediction_df: pd.DataFrame,
                                                treatment: int,
                                                control: int):
    """
    Computes att estimates from predictions of a hydra model, accounting for missing outcomes
    Uses one-step TMLE internally

    Args:
        prediction_df: predictions from hydra model (probably loaded from output of hydra_predict_and_write_to_csv)
        treatment: which treatment level to use as 'treated'
        control: which treatment level to use as 'control'

    Returns: att estimate, att standard deviation estimate

    """
    reduced_df = prediction_df[np.logical_or(prediction_df.treatment == treatment, prediction_df.treatment == control)]

    # condition on either treatment or control being selected
    g0_t = reduced_df['g0_'+str(treatment)]
    g0_c = reduced_df['g0_' + str(control)]
    reduced_df['g0'] = g0_t / (g0_t + g0_c)

    g1_t = reduced_df['g1_'+str(treatment)]
    g1_c = reduced_df['g1_' + str(control)]
    reduced_df['g1'] = g1_t / (g1_t + g1_c)

    # names used by semi-parametric util functions
    selections = {'y': 'outcome',
                  't': 'treatment',
                  'delta': 'outcome_observed',
                  'q0': 'q'+str(control),
                  'q1': 'q'+str(treatment),
                  'p_delta': 'prob_y_obs',
                  'g0': 'g0',
                  'g1': 'g1'}

    reduced_df = reduced_df[selections.values()]
    rename_dict = {v: k for k, v in selections.items()}
    reduced_df = reduced_df.rename(columns=rename_dict)

    # code treatments as 0/1
    is_treatment = reduced_df.t==treatment
    reduced_df.t[is_treatment] = 1
    reduced_df.t[np.logical_not(is_treatment)] = 0

    nuisance_dict = reduced_df.to_dict('series')

    att, IC = tmle_missing_outcomes(**nuisance_dict, cross_ent_outcome=True, deps=0.0001)

    att_std = np.std(IC) / np.sqrt(IC.shape[0])

    return att, att_std
