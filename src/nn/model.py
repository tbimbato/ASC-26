"""Embedding + networks for the neural baseline.

EMBEDDING: log-mel spectrogram, 64 bands. Compressing the frequency axis to 64
rows instead of the STFT's 257 leaves less to memorize on a training set this
size. A Schroeder decay curve would be the hand-crafted physics itself, which
would make the learned-vs-physical comparison circular.

NETWORKS: one embedding, four architectures, so the result cannot be pinned on
one particular net. Two axes:
  - capacity: the same conv stack at ~24k / ~94k / ~370k params (width 0.5/1/2)
  - pretraining: an ImageNet-pretrained ResNet18 fine-tuned on the same log-mels
All four consume the identical input, so what varies is the architecture and
the pretraining, not the representation.
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SmallCNN(nn.Module):
    """3 conv blocks -> global average pool -> linear, batch norm + dropout.
    `width` scales the channel stack (base 32/64/128):
      width 0.5 -> 16/32/64,   ~24k params
      width 1.0 -> 32/64/128,  ~94k params
      width 2.0 -> 64/128/256, ~370k params
    Widths are swept rather than picked so the sim2real result is about
    capacity rather than one particular size."""

    def __init__(self, n_classes: int, width: float = 1.0):
        super().__init__()
        c1, c2, c3 = (int(c * width) for c in (32, 64, 128))
        self.net = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(c3, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18Audio(nn.Module):
    """ImageNet-pretrained ResNet18 fine-tuned on the same 1-channel
    log-mels. First conv adapted to 1 channel by summing the pretrained RGB
        kernels, which preserves the filters' scale and orientation selectivity;
    final fc replaced. ~11M params, all fine-tuned: the variable under test is
    the pretrained features, not frozen versus fine-tuned.
    Weights download once (~45 MB) on first use via torchvision."""

    def __init__(self, n_classes: int):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        w = net.conv1.weight.data.sum(dim=1, keepdim=True)  # (64,3,7,7)->(64,1,7,7)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        net.conv1.weight.data = w
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)
