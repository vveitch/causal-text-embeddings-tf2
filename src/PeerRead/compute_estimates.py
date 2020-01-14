import numpy as np
import pandas as pd

from semi_parametric_estimation.att import tmle_missing_outcomes


def att_from_output_tsv(filename, treatment, control):
    df = pd.read_csv(filename, sep='\t')

    reduced_df = df[np.logical_or(df.treatment == treatment, df.treatment == control)]

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


if __name__ == '__main__':
    filename = 'out/predictions.tsv'

    treatment = 9
    control = 10

    att, att_std = att_from_output_tsv(filename, treatment, control)

