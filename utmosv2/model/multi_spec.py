import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utmosv2.dataset._utils import get_dataset_num


class MultiSpecModelV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert cfg.model.multi_spec.backbone == "tf_efficientnetv2_s.in21k_ft_in1k"
        assert cfg.model.multi_spec.atten == True
        assert cfg.model.multi_spec.pool_type == "catavgmax"
        assert cfg.model.multi_spec.num_classes == 1

        self.backbones = nn.ModuleList(
            [
                timm.create_model(
                    "tf_efficientnetv2_s.in21k_ft_in1k",
                    pretrained=True,
                    num_classes=0,
                )
                for _ in range(len(cfg.dataset.specs))
            ]
        )
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1),
            pool_type="catavgmax",
            flatten=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.backbones[0].num_features
            * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True,
        )

        fc_in_features = (
            self.backbones[0].num_features
            * 2
            * 2
        )

        self.fc = nn.Linear(fc_in_features, 1)

        # if cfg.print_config:
        #     print(f"| backbone model: {cfg.model.multi_spec.backbone}")
        #     print(f"| Pooling: {cfg.model.multi_spec.pool_type}")
        #     print(f"| Number of fc input features: {self.fc.in_features}")
        #     print(f"| Number of fc output features: {self.fc.out_features}")

    def forward(self, x):
        x = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        x = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(x)
        ]
        x = [
            sum(
                [
                    x[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(x, dim=3)
        x = self.pooling(x).squeeze(3)
        xt = torch.permute(x, (0, 2, 1))
        y, _ = self.attn(xt, xt, xt)
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(x)
        return x


class MultiSpecExtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert cfg.model.multi_spec.backbone == "tf_efficientnetv2_s.in21k_ft_in1k"
        assert cfg.model.multi_spec.atten == True
        assert cfg.model.multi_spec.pool_type == "catavgmax"
        assert cfg.model.multi_spec.num_classes == 1

        self.backbones = nn.ModuleList(
            [
                timm.create_model(
                    "tf_efficientnetv2_s.in21k_ft_in1k",
                    pretrained=True,
                    num_classes=0,
                )
                for _ in range(len(cfg.dataset.specs))
            ]
        )
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1),
            pool_type="catavgmax",
            flatten=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.backbones[0].num_features
            * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True,
        )

        fc_in_features = (
            self.backbones[0].num_features
            * 2
            * 2
        )

        self.num_dataset = get_dataset_num(cfg)

        self.fc = nn.Linear(
            fc_in_features + self.num_dataset, 1
        )

        # if cfg.print_config:
        #     print(f"| backbone model: {cfg.model.multi_spec.backbone}")
        #     print(f"| Pooling: {cfg.model.multi_spec.pool_type}")
        #     print(f"| Number of fc input features: {self.fc.in_features}")
        #     print(f"| Number of fc output features: {self.fc.out_features}")

    def forward(self, x, d):
        x = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        x = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(x)
        ]
        x = [
            sum(
                [
                    x[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(x, dim=3)
        x = self.pooling(x).squeeze(3)
        xt = torch.permute(x, (0, 2, 1))
        y, _ = self.attn(xt, xt, xt)
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x
