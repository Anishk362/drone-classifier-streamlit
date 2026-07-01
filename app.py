"""
🛸 Acoustic Drone Detection & Classification v2
Multi-Label Detection — Identifies multiple drones simultaneously

Streamlit app with dark theme UI, waveform / gammatone / MS-WST visualizations,
and multi-label checklist results.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from inference import load_model, classify_audio, SR, DRONE_NAMES, DETECTION_THRESHOLD

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drone Classifier v2",
    page_icon="🛸",
    layout="wide",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00d4ff !important;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #00d4ff;
        font-size: 1.5rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4a5568;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Success / info boxes */
    .stSuccess {
        background-color: #1a3a2a;
        border-left: 4px solid #00d4ff;
    }

    /* Detection result cards */
    .drone-detected {
        background-color: #1a3a2a;
        border-left: 4px solid #22c55e;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .drone-not-detected {
        background-color: #1e1e2e;
        border-left: 4px solid #4a5568;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        color: #9ca3af;
    }
    .drone-background {
        background-color: #2a1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛸 Drone Classifier v2")
    st.markdown("---")

    st.markdown("### Architecture")
    st.markdown("""
    - **Input**: WAV audio (mono, 16 kHz)
    - **Branch 1**: Multi-Scale WST (4 scales)
    - **Branch 2**: Gammatone spectrogram (64 filters)
    - **Fusion**: Scaled Dot-product Cross-Attention
    - **Output**: Sigmoid per class (multi-label)
    """)

    st.markdown("---")
    st.markdown("### Drone Classes")
    st.markdown("""
    | # | Drone |
    |---|-------|
    | 0 | DJI Mavic Mini |
    | 1 | DJI Mavic 3 Cine |
    | 2 | Agricultural Drone |
    """)

    st.markdown("---")
    st.markdown("### Detection Logic")
    st.markdown(f"""
    - Each class scored independently via **sigmoid**
    - Detection threshold: **{DETECTION_THRESHOLD * 100:.0f}%**
    - **Background** = no drone above threshold
    - Multiple drones can be detected simultaneously
    """)

    st.markdown("---")
    st.caption("v2.0 — Multi-label Detector · IIT Jammu")

# ── Main Area ──────────────────────────────────────────────────────────────────

st.markdown("# 🛸 Acoustic Drone Detection & Classification v2")
st.markdown("#### Multi-Label Detection — Identifies multiple drones simultaneously")
st.markdown("---")

# ── Load Model ─────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_model():
    return load_model("best_model_v2.pth")

model = _load_model()
st.success("✅ Model loaded successfully (Multi-label, 3-class detector)")

# ── File Upload ────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav", "mp3", "flac", "ogg"],
    help="Supported formats: WAV, MP3, FLAC, OGG. Audio will be resampled to 16 kHz mono.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔄 Processing audio — extracting features & classifying..."):
        audio_bytes = uploaded_file.getvalue()
        detected_drones, is_background, all_scores, mswst, gamma, y_raw = classify_audio(
            audio_bytes, model
        )

    st.markdown("---")

    # ── Results Section ────────────────────────────────────────────────────────

    st.markdown("## 🎯 Detection Results")

    if is_background:
        st.markdown(
            '<div class="drone-background">🔇 <strong>No drone detected</strong> — Background / Silence</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(detected_drones)} drone(s) detected:**")

    # Show checklist for all classes
    for idx in sorted(DRONE_NAMES.keys()):
        name = DRONE_NAMES[idx]
        score = all_scores.get(name, 0.0)
        pct = score * 100

        is_detected = score >= DETECTION_THRESHOLD
        if is_detected:
            st.markdown(
                f'<div class="drone-detected">✅ <strong>{name}</strong> — {pct:.0f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="drone-not-detected">❌ {name} — {pct:.0f}%</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Bar Chart: All Scores with Threshold Line ──────────────────────────────

    st.markdown("## 📊 Per-Class Detection Scores")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
    fig_bar.patch.set_facecolor("#0e1117")
    ax_bar.set_facecolor("#0e1117")

    names = list(all_scores.keys())
    scores = [all_scores[n] * 100 for n in names]
    colors = ["#22c55e" if s >= DETECTION_THRESHOLD * 100 else "#4a5568" for s in scores]

    bars = ax_bar.barh(names, scores, color=colors, height=0.5, edgecolor="none")

    # Threshold line
    ax_bar.axvline(
        x=DETECTION_THRESHOLD * 100,
        color="#00d4ff",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({DETECTION_THRESHOLD * 100:.0f}%)",
    )

    # Score labels on bars
    for bar, score in zip(bars, scores):
        ax_bar.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            ha="left",
            color="#fafafa",
            fontsize=10,
            fontweight="bold",
        )

    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Confidence (%)", color="#9ca3af", fontsize=10)
    ax_bar.tick_params(colors="#9ca3af", labelsize=9)
    ax_bar.legend(loc="lower right", facecolor="#161b22", edgecolor="#4a5568",
                  labelcolor="#fafafa", fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_color("#4a5568")
    ax_bar.spines["left"].set_color("#4a5568")
    ax_bar.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.markdown("---")

    # ── Visualizations ─────────────────────────────────────────────────────────

    st.markdown("## 📈 Feature Visualizations")

    col1, col2 = st.columns(2)

    # ── Waveform ───────────────────────────────────────────────────────────────
    with col1:
        st.markdown("### 🎵 Waveform")
        fig_wave, ax_wave = plt.subplots(figsize=(6, 2.5))
        fig_wave.patch.set_facecolor("#0e1117")
        ax_wave.set_facecolor("#0e1117")

        t = np.arange(len(y_raw)) / SR
        ax_wave.plot(t, y_raw, color="#00d4ff", linewidth=0.4, alpha=0.85)
        ax_wave.set_xlabel("Time (s)", color="#9ca3af", fontsize=9)
        ax_wave.set_ylabel("Amplitude", color="#9ca3af", fontsize=9)
        ax_wave.tick_params(colors="#9ca3af", labelsize=8)
        ax_wave.set_xlim(0, t[-1] if len(t) > 0 else 1)
        ax_wave.spines["top"].set_visible(False)
        ax_wave.spines["right"].set_visible(False)
        ax_wave.spines["bottom"].set_color("#4a5568")
        ax_wave.spines["left"].set_color("#4a5568")
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close(fig_wave)

    # ── Gammatone Spectrogram ──────────────────────────────────────────────────
    with col2:
        st.markdown("### 🎹 Gammatone Spectrogram")
        fig_gamma, ax_gamma = plt.subplots(figsize=(6, 2.5))
        fig_gamma.patch.set_facecolor("#0e1117")
        ax_gamma.set_facecolor("#0e1117")

        im = ax_gamma.imshow(
            gamma,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="bilinear",
        )
        ax_gamma.set_xlabel("Time Frame", color="#9ca3af", fontsize=9)
        ax_gamma.set_ylabel("Filter #", color="#9ca3af", fontsize=9)
        ax_gamma.tick_params(colors="#9ca3af", labelsize=8)
        ax_gamma.spines["top"].set_visible(False)
        ax_gamma.spines["right"].set_visible(False)
        ax_gamma.spines["bottom"].set_color("#4a5568")
        ax_gamma.spines["left"].set_color("#4a5568")
        cbar = fig_gamma.colorbar(im, ax=ax_gamma, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#9ca3af", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_gamma)
        plt.close(fig_gamma)

    # ── MS-WST Heatmap ─────────────────────────────────────────────────────────
    st.markdown("### 🌊 Multi-Scale Wavelet Scattering Transform (MS-WST)")
    fig_wst, ax_wst = plt.subplots(figsize=(10, 2.5))
    fig_wst.patch.set_facecolor("#0e1117")
    ax_wst.set_facecolor("#0e1117")

    im_wst = ax_wst.imshow(
        mswst,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
    )
    ax_wst.set_xlabel("Time Frame", color="#9ca3af", fontsize=9)
    ax_wst.set_ylabel("Scale", color="#9ca3af", fontsize=9)
    ax_wst.set_yticks(range(4))
    ax_wst.set_yticklabels(["J4/Q8", "J6/Q12", "J7/Q12", "J7/Q16"], fontsize=8)
    ax_wst.tick_params(colors="#9ca3af", labelsize=8)
    ax_wst.spines["top"].set_visible(False)
    ax_wst.spines["right"].set_visible(False)
    ax_wst.spines["bottom"].set_color("#4a5568")
    ax_wst.spines["left"].set_color("#4a5568")
    cbar_wst = fig_wst.colorbar(im_wst, ax=ax_wst, fraction=0.046, pad=0.02)
    cbar_wst.ax.tick_params(colors="#9ca3af", labelsize=7)
    plt.tight_layout()
    st.pyplot(fig_wst)
    plt.close(fig_wst)

else:
    # Landing state
    st.markdown("---")
    st.info("👆 Upload a WAV audio file above to start drone detection.")

    st.markdown("### How it works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 1️⃣ Upload")
        st.markdown("Upload a WAV/MP3/FLAC file containing potential drone audio.")
    with col_b:
        st.markdown("#### 2️⃣ Process")
        st.markdown("Audio is preprocessed with Butterworth filtering, MS-WST, and Gammatone features.")
    with col_c:
        st.markdown("#### 3️⃣ Detect")
        st.markdown("Multi-label classifier identifies which drones (if any) are present simultaneously.")
