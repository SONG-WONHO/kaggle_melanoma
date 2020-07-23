import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # meta features
        meta_emb = 32 + 128 + 32
        self.sex_emb = nn.Embedding(2, 32)
        self.site_emb = nn.Embedding(7, 128)
        self.age_emb = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Linear(1, 32),)

        # image
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

        self.out = nn.Linear(in_features=self.c + meta_emb, out_features=config.num_targets, bias=True)
        self.sub_1 = nn.Linear(in_features=self.c + meta_emb, out_features=3, bias=True)

    def forward(self, x, meta):
        sex, site, age = meta

        sex_emb = self.sex_emb(sex)
        site_emb = self.site_emb(site)
        age_emb = self.age_emb(age.unsqueeze(-1).type(torch.float32))

        meta_feat = torch.cat([sex_emb, site_emb, age_emb], dim=-1)

        # features
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, self.c)
        feat = torch.cat([feat, meta_feat], dim=-1)

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
