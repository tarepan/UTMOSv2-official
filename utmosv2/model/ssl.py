"""SSL-based waveform encoders"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from utmosv2.dataset._utils import get_dataset_num

# `transformers` library's AutoClass name
SSL_NAME = "facebook/wav2vec2-base"
HIDDEN_NUM, IN_FEATURES = 13, 768


class _SSLEncoder(nn.Module):
    """wav2vec 2.0 SSL wave encoder"""

    def __init__(self, sr: int, freeze: bool):
        super().__init__()

        self.sr = sr

        # https://huggingface.co/docs/transformers/v4.43.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        self.processor = AutoFeatureExtractor.from_pretrained(SSL_NAME)
        # https://huggingface.co/docs/transformers/v4.43.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model
        self.model = AutoModel.from_pretrained(SSL_NAME)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """wave-to-"""
        x = self.processor([t.cpu().numpy() for t in x], sampling_rate=self.sr, return_tensors="pt").to(self.model.device)
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

        self.encoder = _SSLEncoder(cfg.sr, cfg.model.ssl.freeze)
        self.weights = nn.Parameter(F.softmax(torch.randn(HIDDEN_NUM), dim=0))
        self.attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=IN_FEATURES, num_heads=8, dropout=0.2, batch_first=True)])
        self.fc = nn.Linear(IN_FEATURES * 2 + get_dataset_num(cfg), 1)

    def forward(self, x, d):
        x = self.encoder(x)
        x = sum([t * w for t, w in zip(x, self.weights)])
        y = x
        y, _ = self.attn[0](y, y, y)  # NOTE: len(self.attn) is 1
        x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        return x
