import pandas as pd
import torch

from utmosv2.dataset import MultiSpecDataset, SSLExtDataset


class SSLLMultiSpecExtDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str, transform=None):
        self.data = data
        self.ssl = SSLExtDataset(cfg, data, phase)
        self.multi_spec = MultiSpecDataset(cfg, data, phase, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            wave:   NDArray (T=t,) - Fixed-length resampled waveform
            spec:   Tensor         -
            ds_idx: Tensor  (D=d,) - Dataset ID onehot vector
            target: Tensor         -
        """

        wave, ds_idx, target = self.ssl[idx]
        spec, _ = self.multi_spec[idx]

        return wave, spec, ds_idx, target
