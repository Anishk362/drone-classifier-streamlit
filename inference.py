# ========================= INFERENCE PIPELINE =========================
# Replicates the EXACT preprocessing → feature extraction pipeline
# from v1_4class_model (preprocess.py + features.py) for real-time use.
# Audio goes in, prediction comes out.

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.signal import butter, filtfilt
from kymatio.torch import Scattering1D
from gammatone.gtgram import gtgram

from model import DroneFullModel

# ---- Audio Config (from preprocess.py) ----
SR           = 16000
FRAME_LEN_MS = 100
FRAME_OVERLAP = 0.5
FILTER_ORDER = 4
HIGHCUT      = 2000

# ---- Feature Config (from features.py) ----
CHUNK_SEC   = 4
TARGET_T    = 128
N_FILTERS   = 64
F_MIN       = 50
WINDOW_TIME = 0.025
HOP_TIME    = 0.010

MS_WST_CONFIGS = [
    {"J": 4, "Q": 8},
    {"J": 6, "Q": 12},
    {"J": 7, "Q": 12},
    {"J": 7, "Q": 16},
]

# ---- Class Labels ----
CLASS_NAMES = {
    0: "Background (No Drone)",
    1: "DJI Mavic Mini",
    2: "DJI Mavic 3 Cine",
    3: "Agricultural Drone",
}

# ---- Initialize Scattering Transforms (CPU) ----
DEVICE = "cpu"
SCATTERINGS = [
    Scattering1D(J=cfg["J"], Q=cfg["Q"], shape=SR * CHUNK_SEC).to(DEVICE)
    for cfg in MS_WST_CONFIGS
]


# ======================= PREPROCESSING =======================
# Exact replication of v1/preprocess.py pipeline

def load_audio(file_path_or_bytes, sr=SR):
    """Load audio from file path or bytes, resample to 16kHz mono."""
    y, _ = librosa.load(file_path_or_bytes, sr=sr, mono=True)
    return y


def peak_normalize(x):
    """Peak normalization to [0, 1] range."""
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin == 0:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def butter_lowpass_filter(x, sr=SR, highcut=HIGHCUT, order=FILTER_ORDER):
    """4th-order Butterworth low-pass filter at 2 kHz."""
    nyq = sr / 2
    highcut = min(highcut, 0.99 * nyq)
    b, a = butter(order, highcut, btype="lowpass", fs=sr)
    return filtfilt(b, a, x)


def frame_audio(x, sr=SR, frame_len_ms=FRAME_LEN_MS, overlap=FRAME_OVERLAP):
    """Slice audio into overlapping frames."""
    frame_len = int(sr * frame_len_ms / 1000)
    hop_len   = int(frame_len * (1 - overlap))
    if len(x) < frame_len:
        return np.empty((0, frame_len))
    return librosa.util.frame(x, frame_length=frame_len, hop_length=hop_len).T


def reconstruct_audio(frames):
    """Overlap-add reconstruction from frames (replicates features.py logic)."""
    if frames.shape[0] == 0:
        return np.zeros(SR * CHUNK_SEC, dtype=np.float32)
    frame_len = frames.shape[1]
    hop_len   = frame_len // 2
    y = np.zeros(hop_len * (len(frames) - 1) + frame_len)
    for i, frame in enumerate(frames):
        y[i * hop_len : i * hop_len + frame_len] += frame
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def preprocess_audio(audio_bytes_or_path):
    """Full preprocessing pipeline: load → normalize → filter → frame → reconstruct."""
    y = load_audio(audio_bytes_or_path, sr=SR)
    y = peak_normalize(y)
    y = butter_lowpass_filter(y)
    frames = frame_audio(y)
    y_reconstructed = reconstruct_audio(frames)
    return y_reconstructed, y  # Return both reconstructed and original for visualization


# ======================= FEATURE EXTRACTION =======================
# Exact replication of v1/features.py pipeline

def compute_mswst(y):
    """Multi-Scale Wavelet Scattering Transform → (4, 128) tensor."""
    chunk_len = SR * CHUNK_SEC
    chunks = []
    for i in range(0, len(y), chunk_len):
        chunk = y[i:i + chunk_len]
        if len(chunk) < chunk_len // 2:
            continue
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk)

    if not chunks:
        return np.zeros((4, TARGET_T), dtype=np.float32)

    mswst_all = []
    for scattering in SCATTERINGS:
        channel_chunks = []
        for chunk in chunks:
            x = torch.tensor(chunk).unsqueeze(0).to(DEVICE)
            S = scattering(x)
            S = torch.mean(S, dim=1)
            S = F.interpolate(S.unsqueeze(1), size=TARGET_T,
                              mode="linear", align_corners=False).squeeze()
            channel_chunks.append(S.cpu())
        mswst_all.append(torch.stack(channel_chunks).mean(dim=0).numpy())

    mswst = np.stack(mswst_all, axis=0)
    for c in range(mswst.shape[0]):
        mswst[c] = (mswst[c] - mswst[c].mean()) / (mswst[c].std() + 1e-8)
    return mswst.astype(np.float32)


def compute_gammatone(y):
    """Gammatone filterbank → (64, 128) tensor."""
    G = gtgram(y, SR, WINDOW_TIME, HOP_TIME, N_FILTERS, F_MIN)
    G = np.abs(G) / (np.max(np.abs(G)) + 1e-8)
    G = F.interpolate(torch.tensor(G).unsqueeze(0),
                      size=TARGET_T, mode="linear",
                      align_corners=False).squeeze(0).numpy()
    for i in range(G.shape[0]):
        G[i] = (G[i] - G[i].mean()) / (G[i].std() + 1e-8)
    return G.astype(np.float32)


# ======================= MODEL LOADING =======================

def load_model(model_path="best_model.pth"):
    """Load the trained DroneFullModel from a .pth file."""
    model = DroneFullModel(num_classes=4, c=64)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ======================= FULL INFERENCE =======================

@torch.no_grad()
def classify_audio(audio_bytes_or_path, model):
    """
    End-to-end inference: audio file → class prediction + confidence scores.

    Returns:
        predicted_class (int): 0-3
        class_name (str): Human-readable class name
        confidence (float): Confidence of the predicted class (0-1)
        all_probs (dict): {class_name: probability} for all classes
        mswst_features (np.ndarray): (4, 128) for visualization
        gamma_features (np.ndarray): (64, 128) for visualization
    """
    # Step 1: Preprocess
    y_processed, y_raw = preprocess_audio(audio_bytes_or_path)

    # Step 2: Extract features
    mswst = compute_mswst(y_processed)
    gamma = compute_gammatone(y_processed)

    # Step 3: Convert to tensors and add batch dimension
    w = torch.tensor(mswst, dtype=torch.float32).unsqueeze(0)
    g = torch.tensor(gamma, dtype=torch.float32).unsqueeze(0)

    # Step 4: Forward pass
    logits = model(w, g)
    probs  = F.softmax(logits, dim=1).squeeze().numpy()

    # Step 5: Extract results
    predicted_class = int(np.argmax(probs))
    class_name      = CLASS_NAMES[predicted_class]
    confidence      = float(probs[predicted_class])
    all_probs       = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}

    return predicted_class, class_name, confidence, all_probs, mswst, gamma, y_raw
