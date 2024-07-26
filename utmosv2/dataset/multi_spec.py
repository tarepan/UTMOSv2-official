import librosa
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from utmosv2.dataset._utils import (
    extend_audio,
    get_dataset_map,
    load_audio,
    select_random_start,
)


class MultiSpecDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str, transform=None):
        self.cfg = cfg
        self.data = data
        self.phase = phase
        self.transform = transform

        # NOTE: Same configs are used for all predefined configs
        assert self.cfg.dataset.spec_frames.mixup_inner is True
        for spec_cfg in self.cfg.dataset.specs:
            assert spec_cfg.mode == "melspec"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """
        Returns:
            spec:   (Level*Clip, RGB, Freq, Frame) - multi-window multi-clipping of mel-frequery log-power spectrograms
            target:                                -
        """

        row = self.data.iloc[idx]

        # Prepare spec
        ## Prepare wave
        file = row["file_path"]
        wave = load_audio(self.cfg, file)
        length = int(self.cfg.dataset.spec_frames.frame_sec * self.cfg.sr)
        wave = extend_audio(self.cfg, wave, length, type=self.cfg.dataset.spec_frames.extend)
        ## Convert a wave into multi-level specs :: -> (Level*Clip, RGB, Freq, Frame)
        specs = []
        for _ in range(self.cfg.dataset.spec_frames.num_frames):
            # waveform clipping
            wave1 = select_random_start(wave, length)
            #  specs
            for spec_cfg in self.cfg.dataset.specs:
                # wave-to-spec :: (T,) -> (Freq, Frame)
                spec = _make_melspec(self.cfg, spec_cfg, wave1)

                # mixup_inner :: -> (Freq, Frame)
                wave2 = select_random_start(wave, length)
                spec2 = _make_melspec(self.cfg, spec_cfg, wave2)
                lmd = np.random.beta(self.cfg.dataset.spec_frames.mixup_alpha, self.cfg.dataset.spec_frames.mixup_alpha)
                spec = lmd * spec + (1 - lmd) * spec2

                # Duplication for computer vision architecture's input :: (Freq, Frame) -> (RGB, Freq, Frame)
                spec = np.stack([spec, spec, spec], axis=0)

                # Preprocessing
                spec = torch.tensor(spec, dtype=torch.float32)
                phase = "train" if self.phase == "train" else "valid"
                spec = self.transform[phase](spec)

                specs.append(spec)
        spec = torch.stack(specs).float()

        # Prepare target
        target = row["mos"]
        target = torch.tensor(target, dtype=torch.float32)

        return spec, target


class MultiSpecExtDataset(MultiSpecDataset):
    def __init__(self, cfg, data: pd.DataFrame, phase: str, transform=None):
        super().__init__(cfg, data, phase, transform)
        self.dataset_map = get_dataset_map(cfg)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            spec:   (Level*Clip, RGB, Freq, Frame) - Spectrograms
            ds_idx: (D=d,)                         - Dataset ID onehot vector
            target:                                -
        """

        spec, target = super().__getitem__(idx)

        # Dataset ID onehot vector
        ds_idx = np.zeros(len(self.dataset_map))
        ds_idx[self.dataset_map[self.data.iloc[idx]["dataset"]]] = 1
        ds_idx = torch.tensor(ds_idx, dtype=torch.float32)

        return spec, ds_idx, target


def _make_melspec(cfg, spec_cfg, wave: np.ndarray) -> np.ndarray:
    """Convert waveform into mel-frequery log-power spectrogram :: (T,) -> (Freq, Frame)"""
    spec = librosa.feature.melspectrogram(
        y=wave,
        sr=cfg.sr,
        n_fft=spec_cfg.n_fft,
        hop_length=spec_cfg.hop_length,
        n_mels=spec_cfg.n_mels,
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    if spec_cfg.norm is not None:
        spec = (spec + spec_cfg.norm) / spec_cfg.norm
    return spec
