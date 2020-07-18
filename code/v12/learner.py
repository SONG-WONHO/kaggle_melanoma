import os
import gc
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data import BalanceBatchSampler
from model import get_model


# average meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# loss function
# def loss_func(pred, target):
#     weight = (target == 1).type(torch.int8) + 1
#     return nn.BCEWithLogitsLoss(weight=weight)(pred, target)


def loss_func(pred, target, smoothing=0.0):

    # weight
    weight = (target == 1).type(torch.int8) + 1

    # smooth label
    target_smooth = torch.zeros_like(target)
    print(target_smooth.shape)

    target_smooth.fill_(smoothing)
    target_smooth.scatter_(0, torch.where(target)[0], 1 - smoothing)

    print(target_smooth.shape)

    return nn.BCEWithLogitsLoss(weight=weight)(pred, target_smooth)


def loss_func_sub(pred, target):
    return nn.CrossEntropyLoss()(pred, target)


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.logger = None
        self.name = "model"

    def train(self, trn_data, val_data, model, optimizer, scheduler):

        ### BalanceBatchSampler
        """
        balance_sampler = BalanceBatchSampler(trn_data.df[:, 1], self.config.batch_size)

        train_loader = DataLoader(
            trn_data, batch_sampler=balance_sampler,
            num_workers=self.config.workers, pin_memory=True,
        )
        """

        train_loader = DataLoader(
            trn_data,
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.workers, pin_memory=True,
        )

        valid_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=self.config.workers, pin_memory=True,
        )

        # loger
        logger = self._create_logger()

        # training
        best_metric = 1e-8
        for epoch in range(self.config.num_epochs):
            tr_outputs = self._train_one_epoch(train_loader, model, optimizer, scheduler)
            vl_outputs = self._valid_one_epoch(valid_loader, model)

            # logging
            logger.loc[epoch] = [np.round(v, 4) for v in tr_outputs] +\
                                [np.round(v, 4) for v in vl_outputs] +\
                                [np.round(optimizer.param_groups[0]['lr'], 8)]

            logger.to_csv(os.path.join(self.config.log_path, f'log.{self.name.split(".")[-1]}.csv'))

            # save model
            if best_metric < logger.loc[epoch, 'val_metric']:
                print(f"... From {best_metric:.4f} To {logger.loc[epoch, 'val_metric']:.4f}")
                best_metric = logger.loc[epoch, 'val_metric']
                self.best_model = copy.deepcopy(model)
                name = self.name
                self.name = f"{name}.epoch_{epoch}"
                self.save()
                self.name = f"{name}.best"
                self.save()
                self.name = name

            if self.config.swa and ((epoch + 1) % 4) == 0:
                optimizer.update_swa()

            # scheduler.step(metrics=vl_loss)
            scheduler.step()

        self.logger = logger

        if self.config.swa:
            optimizer.swap_swa_sgd()

            vl_outputs = self._valid_one_epoch(valid_loader, model)
            print(f"\n ***** SWA Score - loss: {vl_outputs[0]:.4f} metric: {vl_outputs[1]:.4f} acc: {vl_outputs[2]:.4f}\n")

            self.best_model = model
            self.name += ".swa"
            self.save()

    def predict(self, tst_data):
        model = self.best_model

        test_loader = DataLoader(
            tst_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=False
        )

        pred_final = []
        sub_1_final = []

        model.eval()

        test_loader = tqdm(test_loader, leave=False)

        for X_batch, _, _, _ in test_loader:
            X_batch = X_batch.to(self.config.device)

            with torch.no_grad():
                preds, p_sub_1 = model(X_batch)

            pred_final.append(preds.detach().cpu())
            sub_1_final.append(p_sub_1.detach().cpu())

        pred_final = torch.cat(pred_final, dim=0)

        sub_1_final = torch.cat(sub_1_final, dim=0)
        sub_1_final = nn.Softmax()(sub_1_final)[:, -1]

        return pred_final, sub_1_final

    def save(self):
        if self.best_model is None:
            print("Must Train before save !")
            return

        torch.save({
            "logger": self.logger,
            "model_state_dict": self.best_model.cpu().state_dict(),
        }, f"{os.path.join(self.config.model_path, self.name)}.pt")

    def load(self, path, name=None):
        ckpt = torch.load(path)
        self.logger = ckpt['logger']
        model_state_dict = ckpt[name]
        model = get_model(self.config)
        try:
            model.load_state_dict(model_state_dict)
            print("... Single GPU (Train)")
        except:
            def strip_module_str(v):
                if v.startswith('module.'):
                    return v[len('module.'):]

            model_state_dict = {strip_module_str(k): v for k, v in model_state_dict.items()}
            model.load_state_dict(model_state_dict)
            print("... Multi GPU (Train)")

        self.best_model = model.to(self.config.device)
        print("... Model Loaded!")

    def _train_one_epoch(self, train_loader, model, optimizer, scheduler):
        losses = AverageMeter()
        losses_sub_1 = AverageMeter()

        model.train()

        train_iterator = tqdm(train_loader, leave=False)
        for X_batch, y_batch, y_sub_1, _ in train_iterator:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device).type(torch.float32)
            y_sub_1 = y_sub_1.to(self.config.device)

            batch_size = X_batch.size(0)

            preds, p_sub_1 = model(X_batch)

            loss = loss_func(preds.view(-1), y_batch.view(-1))
            losses.update(loss.item(), batch_size)

            loss_sub_1 = loss_func_sub(p_sub_1, y_sub_1.view(-1))
            losses_sub_1.update(loss_sub_1.item(), batch_size)

            optimizer.zero_grad()
            (loss + loss_sub_1).backward()
            optimizer.step()

            train_iterator.set_description(
                f"train bce:{losses.avg:.4f}, sub 1: {losses_sub_1.avg:.4f}, lr:{optimizer.param_groups[0]['lr']:.6f}")

        return losses.avg, losses_sub_1.avg

    def _valid_one_epoch(self, valid_loader, model):
        losses = AverageMeter()
        losses_sub_1 = AverageMeter()
        true_final, pred_final = [], []
        sub_1_final = []

        model.eval()

        valid_loader = tqdm(valid_loader, leave=False)
        for i, (X_batch, y_batch, y_sub_1, _) in enumerate(valid_loader):
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device).type(torch.float32)
            y_sub_1 = y_sub_1.to(self.config.device)

            batch_size = X_batch.size(0)

            with torch.no_grad():
                preds, p_sub_1 = model(X_batch)
                loss = loss_func(preds.view(-1), y_batch.view(-1))
                losses.update(loss.item(), batch_size)

                loss_sub_1 = loss_func_sub(p_sub_1, y_sub_1.view(-1))
                losses_sub_1.update(loss_sub_1.item(), batch_size)

            true_final.append(y_batch.cpu())
            pred_final.append(preds.detach().cpu())
            sub_1_final.append(p_sub_1.detach().cpu())

            valid_loader.set_description(f"valid bce:{losses.avg:.4f}, sub 1: {losses_sub_1.avg:.4f}")

        true_final = torch.cat(true_final, dim=0)
        pred_final = torch.cat(pred_final, dim=0).view(-1)
        pred_final = torch.sigmoid(pred_final)

        vl_score = roc_auc_score(true_final.cpu().numpy(), pred_final.cpu().numpy())
        vl_acc = accuracy_score(true_final.cpu().numpy(), np.round(pred_final.cpu().numpy()))

        # sub
        sub_1_final = torch.cat(sub_1_final, dim=0)
        sub_1_final = nn.Softmax()(sub_1_final)[:, -1]
        sub_1_score = roc_auc_score(true_final.cpu().numpy(), sub_1_final.cpu().numpy())

        # ensemble - original, sub_1
        en_1 = (pred_final.cpu().numpy() + sub_1_final.cpu().numpy()) / 2
        en_1_score = roc_auc_score(true_final.cpu().numpy(), en_1)

        return (
            losses.avg, vl_score, vl_acc,
            losses_sub_1.avg, sub_1_score,
            en_1_score
        )

    def _create_logger(self):
        log_cols = ['tr_loss', 'tr_loss_sub_1',
                    'val_loss', 'val_metric', 'val_acc',
                    'val_loss_sub_1', 'sub_1_score',
                    'en1',
                    'lr']
        return pd.DataFrame(index=range(self.config.num_epochs), columns=log_cols)

    def _cal_metrics(self, pred, true):
        return alaska_weighted_auc(true, pred)
