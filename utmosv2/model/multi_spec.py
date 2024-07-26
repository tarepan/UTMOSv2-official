"""Dataset for spec encoders"""

import timm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from utmosv2.dataset._utils import get_dataset_num


class MultiSpecModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_backbones = len(cfg.dataset.specs)

        assert cfg.model.multi_spec.backbone == "tf_efficientnetv2_s.in21k_ft_in1k"
        assert cfg.model.multi_spec.atten == True
        assert cfg.model.multi_spec.pool_type == "catavgmax"
        assert cfg.model.multi_spec.num_classes == 1

        # EfficientNet-v2: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k
        self.backbones = nn.ModuleList([
            timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
            for _ in range(self.n_backbones)
        ])
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(F.softmax(torch.randn(self.n_backbones), dim=0))
        # SelectAdaptivePool2d: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/adaptive_avgmax_pool.py#L124
        self.pooling = timm.layers.SelectAdaptivePool2d(output_size=(None, 1), pool_type="catavgmax", flatten=False)
        self.attn = nn.MultiheadAttention(embed_dim=self.backbones[0].num_features * 2, num_heads=8, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.backbones[0].num_features * 2 * 2, 1)

    def forward(self, x):
        """
        Args:
            x      :: (B, Level*Clip, RGB, Freq, Frame) - Spectrograms
        Returns:
                   :: (B, 1)                            - Scores
        """
        x = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(self.cfg.dataset.spec_frames.num_frames * self.n_backbones)
        ]
        x = [self.backbones[i % self.n_backbones](specs) for i, specs in enumerate(x)]
        x = [
            sum([x[i * self.n_backbones + j] * w for j, w in enumerate(self.weights)])
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(x, dim=3)
        x = self.pooling(x).squeeze(3)

        # unitSeries-to-feat :: (B, Feat, Frame) -> (B, Frame, Feat) -> (B, Frame, Feat) then (B, Frame, Feat) & (B, Feat, Frame) -> (B, Feat)
        xt = torch.permute(x, (0, 2, 1))
        y, _ = self.attn(xt, xt, xt)
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)

        # feat-to-score :: (B, Feat) -> (B, 1)
        x = self.fc(x)

        return x


class MultiSpecExtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.n_backbones = len(cfg.dataset.specs)

        assert cfg.model.multi_spec.backbone == "tf_efficientnetv2_s.in21k_ft_in1k"
        assert cfg.model.multi_spec.atten == True
        assert cfg.model.multi_spec.pool_type == "catavgmax"
        assert cfg.model.multi_spec.num_classes == 1

        # EfficientNet-v2: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k
        self.backbones = nn.ModuleList([
            timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=0)
            for _ in range(self.n_backbones)
        ])
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(F.softmax(torch.randn(self.n_backbones), dim=0))
        # SelectAdaptivePool2d: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/adaptive_avgmax_pool.py#L124
        self.pooling = timm.layers.SelectAdaptivePool2d(output_size=(None, 1), pool_type="catavgmax", flatten=False)
        self.attn = nn.MultiheadAttention(embed_dim=self.backbones[0].num_features * 2, num_heads=8, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.backbones[0].num_features * 2 * 2 + get_dataset_num(cfg), 1)

    def forward(self, x, ds_idc: Tensor) -> Tensor:
        """
        Args:
            x      :: (B, Level*Clip, RGB, Freq, Frame) - Spectrograms
            ds_idc :: (B, D)                            - Dataset ID onehot vectors
        Returns:
                   :: (B, 1)                            - Scores
        """

        # :: (B, Level*Clip, RGB, Freq, Frame) -> [(B, RGB, Freq, Frame) x Level*Clip]
        x = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(self.cfg.dataset.spec_frames.num_frames * self.n_backbones)
        ]
        # spec-to-feat :: [(B, RGB, Freq, Frame) x Level*Clip] -> [(B, ...) x Level*Clip]
        x = [self.backbones[i % self.n_backbones](specs) for i, specs in enumerate(x)]
        # :: [(B, ...) x Level*Clip] -> [(B, ...) x Clip]
        x = [
            sum([x[i * self.n_backbones + j] * w for j, w in enumerate(self.weights)])
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        # :: [(B, ...) x Clip] -> (B, ..., Clip) -> (B, ...)
        x = torch.cat(x, dim=3)
        x = self.pooling(x).squeeze(3)

        # unitSeries-to-feat :: (B, Feat, Frame) -> (B, Frame, Feat) -> (B, Frame, Feat) then (B, Frame, Feat) & (B, Feat, Frame) -> (B, Feat)
        xt = torch.permute(x, (0, 2, 1))
        y, _ = self.attn(xt, xt, xt)
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)

        # feat-to-score with dataset index :: (B, Feat) & (B, D) -> (B, 1)
        x = self.fc(torch.cat([x, ds_idc], dim=1))

        return x
