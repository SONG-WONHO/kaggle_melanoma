import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLRWarmup(_LRScheduler):
    """
    T_min: epochs for warm-up (ex. T_min=5)
    T_max: epochs for learning including warm-up (ex. T_max=100)
    """
    def __init__(self, optimizer, T_min, T_max, logger=None, eta_min=0, last_epoch=-1, min_lr=1e-8, eps=1e-8):
        self.optimizer = optimizer
        self.T_min = T_min
        self.T_max = T_max
        self.logger = logger
        self.eta_min = eta_min
        self.eps = eps

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        super(CosineAnnealingLRWarmup, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None, verbose=True):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch < self.T_min:
            new_lrs = [base_lr * ((self.last_epoch + 1) / self.T_min) for base_lr in self.base_lrs]
        else:
            new_lrs = [self.eta_min + (base_lr - self.eta_min) *
                       (1 + math.cos(math.pi * ((self.last_epoch + 1) - self.T_min) / (self.T_max - self.T_min + 1))) / 2
                       for base_lr in self.base_lrs]

        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = max(new_lrs[i], self.min_lrs[i])
            param_group['lr'] = new_lr
            if verbose:
                if self.logger is not None:
                    self.logger.log_string('\n\t## Current LR: {:.8f}'.format(new_lr))