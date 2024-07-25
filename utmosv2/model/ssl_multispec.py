"""Fusion models"""

import torch
import torch.nn as nn

from utmosv2.dataset._utils import get_dataset_num
from utmosv2.model import MultiSpecExtModel, MultiSpecModelV2, SSLExtModel


class SSLMultiSpecExtModelV1(nn.Module):
    """Fusion MOS predictor, without dataset vector input to spec sub-model."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.ssl = SSLExtModel(cfg)
        self.spec_long = MultiSpecModelV2(cfg)

        # Load states
        self.ssl.load_state_dict(torch.load(f"outputs/{cfg.model.ssl_spec.ssl_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"))
        self.spec_long.load_state_dict(torch.load(f"outputs/{cfg.model.ssl_spec.spec_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"))

        # Freeze weights
        if cfg.model.ssl_spec.freeze:
            for param in self.ssl.parameters():
                param.requires_grad = False
            for param in self.spec_long.parameters():
                param.requires_grad = False

        # Delete sub-module final FC layers
        ssl_input = self.ssl.fc.in_features
        spec_long_input = self.spec_long.fc.in_features
        self.ssl.fc = nn.Identity()
        self.spec_long.fc = nn.Identity()

        self.num_dataset = get_dataset_num(cfg)
        self.fc = nn.Linear(ssl_input + spec_long_input + self.num_dataset, cfg.model.ssl_spec.num_classes)

    def forward(self, waves, specs, ds_idc):
        """
        Args:
            waves  :: (B, T) - Fixed-length resampled waveforms
            specs  ::        -
            ds_idc :: (B, D) - Dataset ID onehot vectors
        """

        x1 = self.ssl(waves, torch.zeros(waves.shape[0], self.num_dataset).to(waves.device))
        x2 = self.spec_long(specs)
        x = torch.cat([x1, x2, ds_idc], dim=1)
        x = self.fc(x)
        return x


class SSLMultiSpecExtModelV2(nn.Module):
    """UTMOSv2 default MOS predictor."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.ssl = SSLExtModel(cfg)
        self.spec_long = MultiSpecExtModel(cfg)

        # Load module states for training
        if cfg.model.ssl_spec.ssl_weight is not None and cfg.phase == "train":
            self.ssl.load_state_dict(torch.load(f"outputs/{cfg.model.ssl_spec.ssl_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"))
        if cfg.model.ssl_spec.spec_weight is not None and cfg.phase == "train":
            self.spec_long.load_state_dict(torch.load(f"outputs/{cfg.model.ssl_spec.spec_weight}/fold{cfg.now_fold}_s{cfg.split.seed}_best_model.pth"))

        # Freeze weights
        if cfg.model.ssl_spec.freeze:
            for param in self.ssl.parameters():
                param.requires_grad = False
            for param in self.spec_long.parameters():
                param.requires_grad = False

        # Delete sub-module final FC layers
        ssl_input = self.ssl.fc.in_features
        spec_long_input = self.spec_long.fc.in_features
        self.ssl.fc = nn.Identity()
        self.spec_long.fc = nn.Identity()

        self.num_dataset = get_dataset_num(cfg)
        self.fc = nn.Linear(ssl_input + spec_long_input + self.num_dataset, cfg.model.ssl_spec.num_classes)

    def forward(self, waves, specs, ds_idc):
        """
        Args:
            waves  :: (B, T) - Fixed-length resampled waveforms
            specs  ::        -
            ds_idc :: (B, D) - Dataset ID onehot vectors
        """

        x1 = self.ssl(waves, torch.zeros(waves.shape[0], self.num_dataset).to(waves.device))
        x2 = self.spec_long(specs, torch.zeros(x1.shape[0], self.num_dataset).to(x1.device))
        x = torch.cat([x1, x2, ds_idc], dim=1)
        x = self.fc(x)
        return x
