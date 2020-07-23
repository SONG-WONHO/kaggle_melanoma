import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained(config.backbone_name)
        self.c = {
            'efficientnet-b0': 1280,
            'efficientnet-b1': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2560}[config.backbone_name]
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(in_features=self.c, out_features=config.num_targets, bias=True)
        self.sub_1 = nn.Linear(in_features=self.c, out_features=3, bias=True)

    def forward(self, x):
        # features
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, self.c)

        # original outputs
        outputs = self.out(self.dropout(feat))

        # sub outputs
        outputs_sub1 = self.sub_1(self.dropout(feat))

        return outputs, outputs_sub1


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p).reshape(-1, 1280)


class PoolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained(config.backbone_name)
        self.c = {
            'efficientnet-b0': 1280,
            'efficientnet-b1': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2560}[config.backbone_name]
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(in_features=self.c, out_features=config.num_targets, bias=True)
        self.sub_1 = nn.Linear(in_features=self.c, out_features=3, bias=True)
        self.pool_layer = GeM()

    def forward(self, x):
        # features
        feat = self.model.extract_features(x)
        feat = self.pool_layer(feat)


        # original outputs
        outputs = self.out(self.dropout(feat))

        # sub outputs
        outputs_sub1 = self.sub_1(self.dropout(feat))

        return outputs, outputs_sub1


class HeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained(config.backbone_name)
        self.c = {
            'efficientnet-b0': 1280,
            'efficientnet-b1': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2560}[config.backbone_name]
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Sequential(
            nn.Linear(in_features=self.c, out_features=self.c, bias=True),
            nn.LayerNorm(self.c),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.c, out_features=config.num_targets, bias=True))
        self.sub_1 = nn.Sequential(
            nn.Linear(in_features=self.c, out_features=self.c, bias=True),
            nn.LayerNorm(self.c),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.c, out_features=3, bias=True))

    def forward(self, x):
        # features
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, self.c)

        # original outputs
        outputs = self.out(self.dropout(feat))

        # sub outputs
        outputs_sub1 = self.sub_1(self.dropout(feat))

        return outputs, outputs_sub1


def get_model(config):
    try:
        f = globals().get(f"{config.model_name}")
        print(f"... Model Info - {config.model_name}")
        print("...", end=" ")
        return f(config)

    except TypeError:
        raise NotImplementedError("model name not matched")
