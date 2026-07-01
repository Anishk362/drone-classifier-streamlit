"""
Drone Classifier v2 — Multi-label Inference Pipeline

Preprocesses audio (Butterworth filter, MS-WST, Gammatone spectrogram)
and runs multi-label classification with sigmoid thresholding.
"""

import io
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

from kymatio.scattering1d.frontend.torch_frontend import ScatteringTorch1D as Scattering1D
from gammatone.gtgram import gtgram

from model import DroneFullModel

# ── Constants ──────────────────────────────────────────────────────────────────

DRONE_NAMES = {0: "DJI Mavic Mini", 1: "DJI Mavic 3 Cine", 2: "Agricultural Drone"}
DETECTION_THRESHOLD = 0.60

SR = 16000
FRAME_LEN_MS = 100
FRAME_OVERLAP = 0.5
FILTER_ORDER = 4
HIGHCUT = 2000

CHUNK_SEC = 4
TARGET_T = 128

# Gammatone parameters
N_FILTERS = 64
F_MIN = 50
WINDOW_TIME = 0.025
HOP_TIME = 0.010

# Multi-Scale WST configs: 4 scales
MS_WST_CONFIGS = [
    {"J": 4, "Q": 8},
    {"J": 6, "Q": 12},
    {"J": 7, "Q": 12},
    {"J": 7, "Q": 16},
]

DEVICE = torch.device("cpu")

# Global cache for Scattering1D operators (built once on import)
_GLOBAL_SCATTERING_OPS = None
def _get_scattering_ops(n_samples):
    global _GLOBAL_SCATTERING_OPS
    if _GLOBAL_SCATTERING_OPS is None:
        ops = []
        for cfg in MS_WST_CONFIGS:
            sc = Scattering1D(J=cfg["J"], shape=(n_samples,), Q=cfg["Q"]).to(DEVICE).float()
            ops.append(sc)
        _GLOBAL_SCATTERING_OPS = ops
    return _GLOBAL_SCATTERING_OPS


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _butter_lowpass(cutoff, fs, order=FILTER_ORDER):
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype="low", output="sos")


def _apply_lowpass(signal, cutoff=HIGHCUT, fs=SR):
    """Apply Butterworth low-pass filter to signal."""
    sos = _butter_lowpass(cutoff, fs)
    return sosfilt(sos, signal)


def _load_audio(audio_bytes_or_path):
    """Load audio from file path or bytes, return mono signal at SR."""
    if isinstance(audio_bytes_or_path, (str, os.PathLike)):
        y, sr_orig = librosa.load(str(audio_bytes_or_path), sr=None, mono=True)
    else:
        buf = io.BytesIO(audio_bytes_or_path)
        y, sr_orig = sf.read(buf, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
    if sr_orig != SR:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
    return y.astype(np.float32)


def _chunk_audio(y, chunk_samples):
    """Split signal into fixed-length chunks, zero-padding the last one."""
    chunks = []
    for start in range(0, len(y), chunk_samples):
        chunk = y[start : start + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
    return chunks


def _compute_gammatone(chunk, sr=SR):
    """Compute gammatone spectrogram for a single chunk.
    
    Matches extract_features.py: peak normalize, F.interpolate resize, z-norm.
    """
    G = gtgram(chunk, sr, WINDOW_TIME, HOP_TIME, N_FILTERS, F_MIN)
    G = np.abs(G) / (np.max(np.abs(G)) + 1e-8)
    # Resize to TARGET_T using F.interpolate (matches training)
    G = torch.nn.functional.interpolate(
        torch.tensor(G).unsqueeze(0), size=TARGET_T,
        mode="linear", align_corners=False
    ).squeeze(0).numpy()
    # Per-channel z-normalization (matches training)
    for i in range(G.shape[0]):
        G[i] = (G[i] - G[i].mean()) / (G[i].std() + 1e-8)
    return G.astype(np.float32)


def _compute_mswst(chunk, sr=SR):
    """Compute Multi-Scale Wavelet Scattering Transform for a single chunk.
    
    Matches extract_features.py: each scale is interpolated to TARGET_T
    before stacking, then z-normalized per channel.
    """
    x_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(DEVICE)
    n_samples = x_tensor.shape[-1]

    scale_rows = []
    scat_ops = _get_scattering_ops(n_samples)
    for scattering in scat_ops:
        with torch.no_grad():
            sx = scattering(x_tensor)  # (1, channels, time)
        # Mean across channel dim → (1, time)
        sx_mean = torch.mean(sx, dim=1)  # (1, time)
        # Interpolate to TARGET_T
        sx_resized = torch.nn.functional.interpolate(
            sx_mean.unsqueeze(1), size=TARGET_T, mode="linear", align_corners=False
        ).squeeze()  # (TARGET_T,)
        scale_rows.append(sx_resized.cpu().numpy())

    # Stack 4 scales → (4, TARGET_T)
    mswst = np.stack(scale_rows, axis=0)

    # Per-channel z-normalization (matches training)
    for c in range(mswst.shape[0]):
        mswst[c] = (mswst[c] - mswst[c].mean()) / (mswst[c].std() + 1e-8)

    return mswst.astype(np.float32)


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(weights_path="best_model_v2.pth"):
    """Load DroneFullModel with 3-class multi-label head."""
    model = DroneFullModel(num_classes=3)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠ Weights file '{weights_path}' not found — using random weights.")
    model.to(DEVICE)
    model.eval()
    return model


# ── Classification ─────────────────────────────────────────────────────────────

def classify_audio(audio_bytes_or_path, model):
    """
    Multi-label drone classification from audio.

    Parameters
    ----------
    audio_bytes_or_path : str, PathLike, or bytes
        Path to a WAV file or raw audio bytes.
    model : DroneFullModel
        Loaded model instance.

    Returns
    -------
    detected_drones : list[dict]
        List of {"name": str, "confidence": float} for drones above threshold.
    is_background : bool
        True if no drone class exceeds the detection threshold.
    all_scores : dict
        Per-class scores, e.g. {"DJI Mavic Mini": 0.91, ...}.
    mswst : np.ndarray
        MS-WST feature map (4, TARGET_T) for visualization.
    gamma : np.ndarray
        Gammatone spectrogram (N_FILTERS, TARGET_T) for visualization.
    y_raw : np.ndarray
        Raw audio waveform (after resampling) for visualization.
    """
    # Load & preprocess (must match extract_features.py pipeline exactly)
    y_raw = _load_audio(audio_bytes_or_path)

    # Peak normalize → lowpass → frame → reconstruct (exact v1 pipeline)
    y_norm = y_raw.copy()
    xmin, xmax = np.min(y_norm), np.max(y_norm)
    if xmax - xmin > 0:
        y_norm = (y_norm - xmin) / (xmax - xmin)
    y_filtered = _apply_lowpass(y_norm)

    # Frame and reconstruct (matches training)
    frame_len = int(SR * FRAME_LEN_MS / 1000)
    hop_len = int(frame_len * (1 - FRAME_OVERLAP))
    if len(y_filtered) >= frame_len:
        frames = librosa.util.frame(y_filtered, frame_length=frame_len, hop_length=hop_len).T
        # Overlap-add reconstruction
        y_recon = np.zeros(hop_len * (len(frames) - 1) + frame_len)
        for fi, frame in enumerate(frames):
            y_recon[fi * hop_len : fi * hop_len + frame_len] += frame
        y_recon = y_recon / (np.max(np.abs(y_recon)) + 1e-8)
        y_filtered = y_recon.astype(np.float32)

    chunk_samples = CHUNK_SEC * SR
    chunks = _chunk_audio(y_filtered, chunk_samples)

    # Process first chunk (or average over chunks)
    all_logits = []
    mswst_vis = None
    gamma_vis = None

    for i, chunk in enumerate(chunks):
        gamma_feat = _compute_gammatone(chunk)
        mswst_feat = _compute_mswst(chunk)

        # Keep first chunk features for visualization
        if i == 0:
            gamma_vis = gamma_feat.copy()
            mswst_vis = mswst_feat.copy()

        # Prepare tensors: (1, C, T)
        w_tensor = torch.from_numpy(mswst_feat).unsqueeze(0).to(DEVICE)
        g_tensor = torch.from_numpy(gamma_feat).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(w_tensor, g_tensor)  # (1, 3)
        all_logits.append(logits)

    # Average logits across chunks, then sigmoid
    avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)  # (1, 3)
    probs = torch.sigmoid(avg_logits).squeeze(0).cpu().numpy()  # (3,)

    # Build results
    all_scores = {}
    detected_drones = []

    for idx, prob in enumerate(probs):
        name = DRONE_NAMES[idx]
        confidence = float(prob)
        all_scores[name] = round(confidence, 4)

        if confidence >= DETECTION_THRESHOLD:
            detected_drones.append({"name": name, "confidence": confidence})

    # Sort detected drones by confidence (descending)
    detected_drones.sort(key=lambda d: d["confidence"], reverse=True)

    is_background = len(detected_drones) == 0

    return detected_drones, is_background, all_scores, mswst_vis, gamma_vis, y_raw
