import os
import sys
import json
import warnings
import argparse
from pprint import pprint

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchcontrib

from utils import *
from data import *
from transform import get_transform
from model import get_model
from scheduler import CosineAnnealingLRWarmup
from learner import Learner

warnings.filterwarnings("ignore")


class CFG:
    # path
    root_path = "./input/"
    log_path = './log/'
    model_path = './model/'

    # model
    model_name = "BaseModel"
    backbone_name = "efficientnet-b0"
    dropout = 0.2
    weight_decay = 0.001

    # train
    batch_size = 64
    learning_rate = 2e-4
    num_epochs = 20

    # etc
    seed = 42
    workers = 1
    num_targets = 1
    debug = False
    n_folds = 5


def main():
    """ main function
    """

    ### header
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--root-path', default=CFG.root_path,
                        help="root path")
    parser.add_argument('--log-path', default=CFG.log_path,
                        help="log path")
    parser.add_argument('--model-path', default=CFG.model_path,
                        help="model path")
    parser.add_argument('--pretrained-path',
                        help='pretrained path')

    # image
    parser.add_argument('--transform-version', default=0, type=int,
                        help="image transform version ex) 0, 1, 2 ...")
    parser.add_argument('--image-size', default=256, type=int,
                        help="image size(256)")

    # model
    parser.add_argument('--model-name', default=CFG.model_name,
                        help=f"model name({CFG.model_name})")
    parser.add_argument('--backbone-name', default=CFG.backbone_name,
                        help=f"backbone name({CFG.backbone_name})")
    parser.add_argument('--dropout', default=CFG.dropout, type=float,
                        help=f"dropout({CFG.dropout})")
    parser.add_argument('--weight-decay', default=CFG.weight_decay, type=float,
                        help=f"weight decay({CFG.weight_decay})")

    # learning
    parser.add_argument('--batch-size', default=CFG.batch_size, type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument('--learning-rate', default=CFG.learning_rate, type=float,
                        help=f"learning rate({CFG.learning_rate})")
    parser.add_argument('--num-epochs', default=CFG.num_epochs, type=int,
                        help=f"number of epochs({CFG.num_epochs})")
    parser.add_argument("--swa",  action="store_true",
                        help="do stochastic weight averaging")

    # etc
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")
    parser.add_argument("--workers", default=CFG.workers, type=int,
                        help=f"number of workers({CFG.workers})")
    parser.add_argument("--debug",  action="store_true",
                        help="debug mode")

    args = parser.parse_args()

    # path
    CFG.root_path = args.root_path
    CFG.model_path = args.model_path
    CFG.log_path = args.log_path
    CFG.pretrained_path = args.pretrained_path

    # image
    CFG.transform_version = args.transform_version
    CFG.image_size = args.image_size

    # model
    CFG.model_name = args.model_name
    CFG.backbone_name = args.backbone_name
    CFG.dropout = args.dropout
    CFG.weight_decay = args.weight_decay

    # learning
    CFG.batch_size = args.batch_size
    CFG.learning_rate = args.learning_rate
    CFG.num_epochs = args.num_epochs
    CFG.swa = args.swa

    # etc
    CFG.seed = args.seed
    CFG.workers = args.workers
    CFG.debug = args.debug

    # get device
    CFG.device = get_device()

    # get version
    _, version, _ = sys.argv[0].split('/')
    CFG.version = version

    # update log path
    if not CFG.debug:
        CFG.log_path = os.path.join(CFG.log_path, CFG.version)
        os.makedirs(CFG.log_path, exist_ok=True)
        CFG.log_path = os.path.join(CFG.log_path, f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
        os.makedirs(CFG.log_path, exist_ok=True)
    else:
        CFG.log_path = os.path.join(CFG.log_path, "debug")
        os.makedirs(CFG.log_path, exist_ok=True)
        CFG.log_path = os.path.join(CFG.log_path, "debug")
        os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    if not CFG.debug:
        CFG.model_path = os.path.join(CFG.model_path, version)
        os.makedirs(CFG.model_path, exist_ok=True)
        CFG.model_path = os.path.join(CFG.model_path, f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
        os.makedirs(CFG.model_path, exist_ok=True)
    else:
        CFG.model_path = os.path.join(CFG.model_path, "debug")
        os.makedirs(CFG.model_path, exist_ok=True)
        CFG.model_path = os.path.join(CFG.model_path, "debug")
        os.makedirs(CFG.model_path, exist_ok=True)

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
    json.dump(
        {k: v for k, v in dict(CFG.__dict__).items() if '__' not in k},
        open(os.path.join(CFG.log_path, 'CFG.json'), "w"))
    print()

    ### seed all
    seed_everything(CFG.seed)

    ### data related
    # load data
    print("Load Raw Data")
    data_df, test_df = load_data(CFG)

    # preprocess data
    print("Preprocess Data")
    data_df = preprocess_data(CFG, data_df)

    # split data into train with valid
    print("Split Data")
    data_df = split_data(CFG, data_df)

    # get transform
    print("Get Transform")
    train_transforms, test_transforms = get_transform(CFG)

    # oof preds
    data_df['preds'] = np.nan
    data_df['preds_sub_1'] = np.nan

    # train test split
    for fold in range(CFG.n_folds):
        print(f"\nValidation Fold: {fold}")
        train_df = data_df[data_df['fold'] != fold].reset_index(drop=True)
        valid_df = data_df[data_df['fold'] == fold].reset_index(drop=True)
        print(f"... Train Shape: {train_df.shape}, Valid Shape: {valid_df.shape}")

        # dataset
        trn_data = MelanomaDataset(CFG, train_df, train_transforms)
        val_data = MelanomaDataset(CFG, valid_df, test_transforms)

        ### model related
        # get learner
        learner = Learner(CFG)
        learner.name = f"model.fold_{fold}"
        if CFG.pretrained_path:
            print("Load Pretrained Model")
            print(f"... Pretrained Info - {CFG.pretrained_path}")
            learner.load(CFG.pretrained_path, f"model_state_dict")

        # get model
        if CFG.pretrained_path:
            print(f"Get Model")
            model = learner.best_model.to(CFG.deivce)

        else:
            print(f"Get Model")
            model = get_model(CFG)
            model = model.to(CFG.device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # get optimizer
        module_names = list(model.state_dict())
        no_decay = ['bias']
        for m in module_names:
            if 'running_mean' in m:
                no_decay.append(m.split('.running_mean')[0])
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': CFG.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        optimizer = optim.AdamW(optimizer_grouped_parameters, CFG.learning_rate)

        # get optimizer
        # optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)

        if CFG.swa:
            optimizer = torchcontrib.optim.SWA(optimizer)

        # get scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', patience=1, verbose=False, factor=0.2)
        # scheduler = CosineAnnealingLRWarmup(optimizer, T_min=int(CFG.num_epochs / 5), T_max=CFG.num_epochs)
        T = 4
        scheduler = CosineAnnealingLRWarmup(optimizer, T_min=0, T_max=CFG.num_epochs//T)

        ### train related
        # train model
        learner.train(trn_data, val_data, model, optimizer, scheduler)

        ### save OOF
        # predict valid
        preds, sub_1 = learner.predict(val_data)
        preds = torch.sigmoid(preds.view(-1)).numpy()
        sub_1 = sub_1.view(-1).numpy()

        data_df.loc[data_df['fold'] == fold, 'preds'] = preds
        data_df.loc[data_df['fold'] == fold, 'preds_sub_1'] = sub_1

        print()

    np.save(f'{os.path.join(CFG.log_path, "preds.npy")}', data_df['preds'].values)
    np.save(f'{os.path.join(CFG.log_path, "sub_1.npy")}', data_df['preds_sub_1'].values)


if __name__ == "__main__":
    main()
