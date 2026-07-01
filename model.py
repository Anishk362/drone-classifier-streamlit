import torch
import torch.nn as nn
import torch.nn.functional as F


class StatPool(nn.Module):
    """Statistical pooling: concatenates mean and std over the time dimension."""
    def forward(self, x):
        return torch.cat([x.mean(2), x.std(2, correction=0)], dim=1)


class WSTDecoder(nn.Module):
    """Decoder branch for Multi-Scale Wavelet Scattering Transform features."""
    def __init__(self, in_ch=4, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c // 2, 5, padding=2), nn.BatchNorm1d(c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        return self.net(x)


class GammaDecoder(nn.Module):
    """Decoder branch for Gammatone spectrogram features."""
    def __init__(self, in_ch=64, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c // 2, 5, padding=2), nn.BatchNorm1d(c // 2), nn.ReLU(),
            nn.Conv1d(c // 2, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        return self.net(x)


class SDCAM(nn.Module):
    """Scaled Dot-product Cross-Attention Module."""
    def __init__(self, c=64):
        super().__init__()
        self.q = nn.Conv1d(c, c, 1)
        self.k = nn.Conv1d(c, c, 1)
        self.v = nn.Conv1d(c, c, 1)
        self.scale = c ** -0.5

    def forward(self, x, y):
        attn = torch.bmm(self.q(x).transpose(1, 2), self.k(y)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, self.v(y).transpose(1, 2))
        return out.transpose(1, 2)


class DroneEmbeddingNet(nn.Module):
    """Dual-branch embedding network with cross-attention fusion."""
    def __init__(self, c=64):
        super().__init__()
        self.wst = WSTDecoder(4, c)
        self.gamma = GammaDecoder(64, c)
        self.cam_w = SDCAM(c)
        self.cam_g = SDCAM(c)
        self.pool = StatPool()
        self.fusion = nn.Sequential(
            nn.Linear(4 * c, 4 * c), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(4 * c, 2 * c), nn.ReLU(), nn.Dropout(0.4),
        )

    def forward(self, w, g):
        zw = self.wst(w)
        zg = self.gamma(g)
        aw = self.pool(self.cam_w(zw, zg))
        ag = self.pool(self.cam_g(zg, zw))
        return self.fusion(torch.cat([aw, ag], dim=1))


class DroneFullModel(nn.Module):
    """
    Full drone classifier model (v2 — multi-label, 3-class).

    Combines the dual-branch embedding network with a classification head.
    Default num_classes=3 for multi-label drone detection.
    """
    def __init__(self, num_classes=3, c=64):
        super().__init__()
        self.embedder = DroneEmbeddingNet(c)
        self.classifier = nn.Sequential(
            nn.Linear(2 * c, c), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(c, c // 2), nn.ReLU(),
            nn.Linear(c // 2, num_classes),
        )

    def forward(self, w, g):
        return self.classifier(self.embedder(w, g))
