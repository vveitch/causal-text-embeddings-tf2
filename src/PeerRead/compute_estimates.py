import pandas as pd

from causal_bert.misc_utils import hydra_att_from_predictions_missing_outcomes

if __name__ == '__main__':
    tsv_file = 'out/predictions.tsv'
    prediction_df = pd.read_csv(tsv_file, sep='\t')


    treatment = 9
    control = 10

    att, att_std = hydra_att_from_predictions_missing_outcomes(prediction_df, treatment, control)

