import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import cv2

from torch.utils.data import Dataset


def load_data(config):
    """load raw data

    Args:
        config: CFG

    Returns: train_df, test_df
    """

    data_path = os.path.join(config.root_path, "melanoma-external-malignant-256")

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df['target'] = np.nan

    train_df['image_path'] = data_path + "/train/" + train_df['image_name'] + ".jpg"
    test_df['image_path'] = data_path + "/test/" + test_df['image_name'] + ".jpg"

    if config.debug:
        train_df = train_df.sample(40)
        test_df = test_df.sample(40)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    print(f"... Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

    return train_df, test_df


def preprocess_data(config, df):
    """ preprocess data

    Args:
        config: CFG
        df: dataframe object

    Returns: preprocessed dataframe object
    """

    df['patient_id'].fillna("NaN", inplace=True)
    return df


def split_data(config, df):
    """split data into training and validation data

    Args:
        config: CFG
        df: dataframe object

    Returns: df: dataframe object with fold
    """

    df['fold'] = -1

    gkf = GroupKFold(n_splits=config.n_folds)
    for fold, (_, vl_idx) in enumerate(gkf.split(X=df, groups=df['patient_id'])):
        df.loc[vl_idx, "fold"] = fold

    return df


class MelanomaDataset(Dataset):
    def __init__(self, config, df, transforms=None):
        self.config = config
        self.df = df.values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        _, _, _, _, _, label, fn, _ = self.df[idx]
        im = cv2.imread(fn)

        # Apply transformations
        if self.transforms:
            # im = self.transforms(image=im)['image']
            im = self.transforms(im)

        return im, label
