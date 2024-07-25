import json

import librosa
import numpy as np


def load_audio(cfg, file: str) -> np.ndarray:
    """Load a resampled mono waveform."""
    if file.suffix == ".wav":
        wave, sr = librosa.load(file, sr=None)
        wave = librosa.resample(wave, orig_sr=sr, target_sr=cfg.sr)
    else:
        wave = np.load(file)
    return wave


def extend_audio(cfg, wave: np.ndarray, length: int, type: str) -> np.ndarray:
    """
    Args:
        wave   :: (T,) - waveform
        length         - Target waveform length
        type           - How to extend
    Returns:
               :: (T,) - waveform which is longer than `length`
    """
    if wave.shape[0] > length:
        return wave
    elif type == "tile":
        n = length // wave.shape[0] + 1
        wave = np.tile(wave, n)
        return wave
    else:
        raise NotImplementedError


def select_random_start(wave: np.ndarray, length: int) -> np.ndarray:
    """Clip waveform into fixed length with random start."""
    start = np.random.randint(0, wave.shape[0] - length)
    return wave[start : start + length]


def get_dataset_map(cfg) -> dict[str, int]:
    """Acquire the dataset ID dictionary."""

    if cfg.data_config:
        with open(cfg.data_config, "r") as f:
            datasets = json.load(f)
        return {d["name"]: i for i, d in enumerate(datasets["data"])}
    else:
        return {
            "bvcc": 0,
            "sarulab": 1,
            "blizzard2008": 2,
            "blizzard2009": 3,
            "blizzard2010-EH1": 4,
            "blizzard2010-EH2": 5,
            "blizzard2010-ES1": 6,
            "blizzard2010-ES3": 7,
            "blizzard2011": 8,
            "somos": 9,
        }


def get_dataset_num(cfg) -> int:
    """Acquire the number of registered datasets."""
    return len(get_dataset_map(cfg))
