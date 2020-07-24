import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Iterator, List, Optional, Union

import cv2
from PIL import Image

from torch.utils.data import Dataset, Sampler


def load_data(config):
    """load raw data

    Args:
        config: CFG

    Returns: train_df, test_df
    """

    # data_path = os.path.join(config.root_path, "melanoma-external-malignant-256")
    data_path = os.path.join(config.root_path, "jpeg-melanoma-256x256")

    train_df = pd.read_csv(os.path.join(data_path, "train_v2.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test_v2.csv"))
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

    # add aux target - anatom_site_general_challenge
    mp = {
        'torso': 0,
        'lower extremity': 1,
        'upper extremity': 2,
        'head/neck': 3,
        'nan': 4,
        'palms/soles': 5,
        'oral/genital': 6
    }
    train_df['aux_1'] = train_df['anatom_site_general_challenge'].astype(str).map(mp)
    test_df['aux_1'] = np.nan

    ### meta features
    # sex
    mp = {'male': 0, 'female': 1}
    train_df['sex_enc'] = train_df['sex'].fillna("male").map(mp)
    test_df['sex_enc'] = test_df['sex'].fillna("male").map(mp)

    # age
    train_df['age_approx'] = np.log1p(train_df['age_approx'].fillna(50).astype("float32"))
    test_df['age_approx'] = np.log1p(test_df['age_approx'].fillna(50).astype("float32"))

    # width
    train_df['width'] = np.log1p(train_df['width'].astype("float32"))
    test_df['width'] = np.log1p(test_df['width'].astype("float32"))

    # height
    train_df['height'] = np.log1p(train_df['height'].astype("float32"))
    test_df['height'] = np.log1p(test_df['height'].astype("float32"))

    # anatom_site_general_challenge
    mp = {
        'head/neck': 0,
        'upper extremity': 1,
        'lower extremity': 2,
        'torso': 3,
        'nan': 4,
        'palms/soles': 5,
        'oral/genital': 6}
    train_df['site_enc'] = train_df['anatom_site_general_challenge'].astype(str).map(mp)
    test_df['site_enc'] = test_df['anatom_site_general_challenge'].astype(str).map(mp)

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
        self.df = df[['image_path', 'target', 'sub_1', 'aux_1', 'sex_enc', 'site_enc']].values
        self.cont = df[['age_approx', 'width', 'height']].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn, label, sub_1, aux_1, sex, site = self.df[idx]
        cont = self.cont[idx]

        im = cv2.imread(fn)

        # Apply transformations
        if self.transforms:
            # if albumentations
            im = self.transforms(image=im)['image']
            # if torch torchvision
            # im = self.transforms(Image.fromarray(im))

        return im, label, sub_1, aux_1, sex, site, cont


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
