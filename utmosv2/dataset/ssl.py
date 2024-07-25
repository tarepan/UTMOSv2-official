"""Dataset for SSL wave encoders"""

import numpy as np
import pandas as pd
import torch

from utmosv2.dataset._utils import (
    extend_audio,
    get_dataset_map,
    load_audio,
    select_random_start,
)


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str):
        self.cfg = cfg
        self.data = data
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            wave:   NDArray (T=t,) - Fixed-length resampled waveform
            target: Tensor         -
        """
        row = self.data.iloc[idx]
        file = row["file_path"]
        wave = load_audio(self.cfg, file)
        length = int(self.cfg.dataset.ssl.duration * self.cfg.sr)
        wave = extend_audio(self.cfg, wave, length, type="tile")
        wave = select_random_start(wave, length)

        target = row["mos"]
        target = torch.tensor(target, dtype=torch.float32)

        return wave, target


class SSLExtDataset(SSLDataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str):
        super().__init__(cfg, data, phase)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx):
        """
        Returns:
            wave:   NDArray (T=t,) - Fixed-length resampled waveform
            ds_idx: Tensor  (D=d,) - Dataset ID onehot vector
            target: Tensor         -
        """

        wave, target = super().__getitem__(idx)

        # Dataset ID onehot vector
        ds_idx = np.zeros(len(self.dataset_map))
        ds_idx[self.dataset_map[self.data.iloc[idx]["dataset"]]] = 1
        ds_idx = torch.tensor(ds_idx, dtype=torch.float32)

        return wave, ds_idx, target
