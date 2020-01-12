import pandas as pd
import numpy as np


def add_splits_to_label_df(label_df_path):
    label_df = pd.read_feather(label_df_path)
    splits = np.random.randint(0, 100, size=label_df.shape[0])
    label_df['many_split'] = splits
    label_df.to_feather(label_df_path)