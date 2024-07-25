from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from utmosv2.dataset._utils import get_dataset_num

# `transformers` library's AutoClass name
SSL_NAME = "facebook/wav2vec2-base"


class _SSLEncoder(nn.Module):
    def __init__(self, sr: int, freeze: bool):
        super().__init__()
        self.sr = sr
        self.processor = AutoFeatureExtractor.from_pretrained(SSL_NAME)
        self.model = AutoModel.from_pretrained(SSL_NAME)
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

        # NOTE: Same SSL configs are used for all predefined configs
        assert cfg.model.ssl.name == SSL_NAME
        assert cfg.model.ssl.attn == 1
        assert cfg.model.ssl.num_classes == 1

        self.encoder = _SSLEncoder(
            cfg.sr, cfg.model.ssl.freeze
        )
        hidden_num, in_features = 13, 768
        self.weights = nn.Parameter(F.softmax(torch.randn(hidden_num), dim=0))
        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=in_features,
                    num_heads=8,
                    dropout=0.2,
                    batch_first=True,
                )
            ]
        )
        self.num_dataset = get_dataset_num(cfg)
        self.fc = nn.Linear(
            in_features * 2 + self.num_dataset, 1
        )

    def forward(self, x, d):
        x = self.encoder(x)
        x = sum([t * w for t, w in zip(x, self.weights)])
        y = x
        y, _ = self.attn[0](y, y, y)  # NOTE: len(self.attn) is 1
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x
