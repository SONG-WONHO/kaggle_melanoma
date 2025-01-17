import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Iterator, List, Optional, Union

import cv2

from torch.utils.data import Dataset, Sampler


def load_data(config):
    """load raw data

    Args:
        config: CFG

    Returns: train_df, test_df
    """

    # data_path = os.path.join(config.root_path, "melanoma-external-malignant-256")
    data_path = os.path.join(config.root_path, "jpeg-melanoma-256x256")

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df['target'] = np.nan

    train_df['image_path'] = data_path + "/train/" + train_df['image_name'] + ".jpg"
    test_df['image_path'] = data_path + "/test/" + test_df['image_name'] + ".jpg"

    # add sub targets 1 - diagnosis
    mp = {
        'unknown': 0, 'seborrheic keratosis': 0, 'lentigo NOS': 0, 'lichenoid keratosis': 0,
        'solar lentigo': 0, 'cafe-au-lait macule': 0, 'atypical melanocytic proliferation': 0,
        'nevus': 1,
        'melanoma': 2,
    }
    train_df['sub_1'] = train_df['diagnosis'].map(mp)
    test_df['sub_1'] = np.nan

    # add sub targets 2 - anatom_site_general_challenge
    train_df['sub_2'] = train_df['anatom_site_general_challenge'].astype(str) + "_" + train_df['sub_1'].astype(str)

    mp = {
        'torso_0': 0, 'lower extremity_0': 1, 'upper extremity_0': 2, 'torso_1': 3, 'head/neck_0': 4,
        'lower extremity_1': 5, 'upper extremity_1': 6, 'nan_0': 6, 'palms/soles_0': 6, 'torso_2': 7,
        'head/neck_1': 6, 'lower extremity_2': 7, 'oral/genital_0': 6, 'upper extremity_2': 7,
        'head/neck_2': 7, 'nan_1': 6, 'nan_2': 7, 'palms/soles_2': 7, 'oral/genital_2': 7, 'palms/soles_1': 6
    }
    train_df['sub_2'] = train_df['sub_2'].map(mp)
    test_df['sub_2'] = np.nan

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
    df['sex'].fillna("NaN", inplace=True)
    return df


def split_data(config, df):
    """split data into training and validation data

    Args:
        config: CFG
        df: dataframe object

    Returns: df: dataframe object with fold
    """

    df['fold'] = -1

    # MultilabelStratifiedKFold
    mskf = MultilabelStratifiedKFold(n_splits=config.n_folds, random_state=config.seed, shuffle=True)

    # patient level - target, sex, size
    patient = df.groupby("patient_id")['target'].apply(lambda v: (v == 1).any())
    patient = pd.concat([patient, df.groupby('patient_id')['sex'].apply(lambda v: v.iloc[0])], axis=1)
    patient = pd.concat([patient, df.groupby("patient_id").size()], axis=1).rename({0: "size"}, axis=1)
    patient['size'] = pd.qcut(patient['size'], 10, labels=range(10))

    for fold, (tr_idx, vl_idx) in enumerate(mskf.split(X=patient, y=patient.values)):
        vl_idx = df[df['patient_id'].isin(patient.iloc[vl_idx].index)].index
        df.loc[vl_idx, "fold"] = fold

    return df


class MelanomaDataset(Dataset):
    def __init__(self, config, df, transforms=None):
        self.config = config
        self.df = df[['image_path', 'target', 'sub_1', 'sub_2']].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn, label, sub_1, sub_2 = self.df[idx]
        im = cv2.imread(fn)

        # Apply transformations
        if self.transforms:
            # if albumentations
            im = self.transforms(image=im)['image']
            # if torch toolbox
            # im = self.transforms(im)

        return im, label, sub_1, sub_2


# noinspection PyMissingConstructor
class BalanceBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle= shuffle

        labels = np.array(labels)

        samples_per_class = {
            label: (labels == label).sum()
            for label
            in set(labels)}

        rate_per_class = {
            label: samples_per_class[label] / len(labels)
            for label
            in set(labels)}

        bs_per_class = {
            label: int(np.ceil(rate_per_class[label] * batch_size))
            for label
            in set(labels)}
        for idx, (label, cnt) in enumerate(sorted(bs_per_class.items(), key=lambda v: v[1])):
            if idx == len(samples_per_class) - 1:
                break

            batch_size -= cnt
        bs_per_class[label] = batch_size

        lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label
            in set(labels)}

        self.samples_per_class = samples_per_class
        self.rate_per_class = rate_per_class
        self.bs_per_class = bs_per_class
        self.lbl2idx = lbl2idx

        self.steps_per_epoch = min(
            *[int(np.ceil(samples_per_class[label] / bs_per_class[label]))
              for label
              in set(labels)])

    def __iter__(self):
        if self.shuffle:
            for label in self.samples_per_class.keys():
                self.lbl2idx[label] = np.random.permutation(self.lbl2idx[label]).tolist()

        batch = []
        for idx in range(self.steps_per_epoch):
            for label in self.samples_per_class.keys():
                batch += self.lbl2idx[label][idx * self.bs_per_class[label]:(idx + 1) * self.bs_per_class[label]]
            yield np.random.permutation(batch).tolist()
            batch = []

    def __len__(self):
        return self.steps_per_epoch
