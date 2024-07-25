"""Datasets"""

from utmosv2.dataset.multi_spec import MultiSpecDataset, MultiSpecExtDataset
from utmosv2.dataset.ssl import SSLDataset, SSLExtDataset
from utmosv2.dataset.ssl_multispec import SSLLMultiSpecExtDataset

__all__ = [
    "MultiSpecDataset",        # Yield [      spec,         target]
    "MultiSpecExtDataset",     # Yield [      spec, ds_idx, target]
    "SSLLMultiSpecExtDataset", # Yield [wave, spec, ds_idx, target]
    "SSLDataset",              # Yield [wave,               target]
    "SSLExtDataset",           # Yield [wave,       ds_idx, target]

]
