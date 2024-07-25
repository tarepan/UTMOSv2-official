from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from utmosv2.dataset._utils import get_dataset_num


class _SSLEncoder(nn.Module):
    def __init__(self, sr: int, model_name: str, freeze: bool):
        super().__init__()
        self.sr = sr
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.processor(
            [t.cpu().numpy() for t in x],
            sampling_rate=self.sr,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model(**x, output_hidden_states=True)
        return outputs.hidden_states


class SSLExtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = _SSLEncoder(
            cfg.sr, cfg.model.ssl.name, cfg.model.ssl.freeze
        )
        hidden_num, in_features = get_ssl_output_shape(cfg.model.ssl.name)
        self.weights = nn.Parameter(F.softmax(torch.randn(hidden_num), dim=0))
        if cfg.model.ssl.attn:
            self.attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=in_features,
                        num_heads=8,
                        dropout=0.2,
                        batch_first=True,
                    )
                    for _ in range(cfg.model.ssl.attn)
                ]
            )
        self.num_dataset = get_dataset_num(cfg)
        self.fc = nn.Linear(
            in_features * 2 + self.num_dataset, cfg.model.ssl.num_classes
        )

    def forward(self, x, d):
        x = self.encoder(x)
        x = sum([t * w for t, w in zip(x, self.weights)])
        if self.cfg.model.ssl.attn:
            y = x
            for attn in self.attn:
                y, _ = attn(y, y, y)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        else:
            x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x


def get_ssl_output_shape(name: str) -> tuple[int, int]:
    if name in [
        "facebook/w2v-bert-2.0",
        "facebook/wav2vec2-large",
        "facebook/wav2vec2-large-robust",
        "facebook/wav2vec2-large-960h",
        "microsoft/wavlm-large",
        "facebook/wav2vec2-large-xlsr-53",
    ]:
        return 25, 1024
    elif name in [
        "facebook/hubert-base-ls960",
        "facebook/data2vec-audio-base-960h",
        "microsoft/wavlm-base",
        "microsoft/wavlm-base-plus",
        "microsoft/wavlm-base-plus-sv",
        "facebook/wav2vec2-base",
    ]:
        return 13, 768
    else:
        raise NotImplementedError
