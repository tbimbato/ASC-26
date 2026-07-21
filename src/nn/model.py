"""Embedding + network for the neural baseline. Both still open questions.

EMBEDDING (what the net sees) -- to reason about, candidates:
  a) log-magnitude STFT spectrogram of the raw IR. Most honest for the
     "learned vs physical features" comparison: no physics pre-cooked in,
     the net has to discover decay/clarity structure by itself.
  b) log-mel spectrogram: same idea, cheaper, perceptual freq axis.
  c) log energy-decay curve (Schroeder): tempting but it IS the physics we
     hand-crafted, would make the comparison circular. Probably no.
  DECIDED (21 July): (b), log-mel with 64 bands. Same picture as the STFT but
  the frequency axis is compressed the way the ear does it, 64 rows instead
  of 257. Smaller input, less to memorize with only 2200 training rooms, and
  still no physics pre-cooked in.

NETWORK -- to reason about, candidates:
  a) small 2-D CNN on the spectrogram (few conv blocks + global pool + linear).
     2200 training rooms is little, keep it tiny or it memorizes.
  b) 1-D CNN on the raw IR. More radical, probably data-hungry.
  Current placeholder: (a), a deliberately small conv stack.
"""

import librosa
import numpy as np
import torch
import torch.nn as nn

from ingest import TARGET_FS

N_FFT = 512
HOP = 128
N_MELS = 64


def embed(ir: np.ndarray) -> np.ndarray:
    """IR (fixed-length 1-D) -> log-mel spectrogram, shape (1, mel, time)."""
    m = librosa.feature.melspectrogram(
        y=ir, sr=TARGET_FS, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return np.log10(m + 1e-8).astype(np.float32)[None, :, :]


class SmallCNN(nn.Module):
    """3 conv blocks -> global average pool -> linear. v2: batch norm for
    stable training, dropout before the classifier against memorization.
    v3: doubled channels (~95k params, was ~24k). There is no magic
    params/dataset ratio; with augmentation anything from tens of k to a few
    hundred k is defensible on 1760 training images, and 24k risked looking
    like a strawman for the "vs neural" comparison."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)
