import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time

from inference import classify_audio, load_model, CLASS_NAMES, SR

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Drone Acoustic Classifier",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= CUSTOM CSS =========================
# Custom CSS removed to fix blank screen issue.


# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("## 🛸 Model Info")
    st.markdown("---")

    st.markdown("**Architecture**")
    st.markdown("""
    `DroneFullModel` — Dual-stream cross-attention pipeline:
    - **WSTDecoder**: 3-layer 1D CNN on MS-WST features
    - **GammaDecoder**: 3-layer 1D CNN on Gammatone features
    - **SDCAM**: Bidirectional cross-attention fusion
    - **StatPool**: Mean + Std temporal pooling
    - **MLP Classifier**: 128 → 64 → 32 → 4
    """)

    st.markdown("---")
    st.markdown("**Training Config**")
    st.markdown("""
    | Param | Value |
    |-------|-------|
    | Optimizer | AdamW |
    | LR | 1e-3 |
    | Dropout | 0.4 |
    | Label Smoothing | 0.1 |
    | Epochs | 75 |
    """)

    st.markdown("---")
    st.markdown("**Performance**")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Accuracy | **98.06%** |
    | Global FAR | 0.68% |
    | Global FRR | 2.36% |
    """)

    st.markdown("---")
    st.markdown("**Classes**")
    for i, name in CLASS_NAMES.items():
        st.markdown(f"**{i}**: {name}")

    st.markdown("---")
    st.caption("IIT Jammu Research Internship 2025")
    st.caption("Anish Kalra | 24MC3006")

# ========================= MAIN CONTENT =========================

# Header
st.markdown("""
<div class="main-header">
    <h1>🛸 Acoustic Drone Detection & Classification</h1>
    <p>Upload an audio file to classify it using the Dual-Feature Cross-Attention Deep Learning Pipeline</p>
</div>
""", unsafe_allow_html=True)

# Load model (cached)
@st.cache_resource
def get_model():
    return load_model("best_model.pth")

try:
    model = get_model()
    st.success("✅ Model loaded successfully (738 KB, 4-class classifier)")
except FileNotFoundError:
    st.error("❌ `best_model.pth` not found! Place it in the same directory as `app.py`.")
    st.stop()

# File Upload
st.markdown("### 📁 Upload Audio File")
uploaded_file = st.file_uploader(
    "Supported formats: WAV, MP3, FLAC, OGG",
    type=["wav", "mp3", "flac", "ogg"],
    help="Upload a drone or background audio recording for classification."
)

if uploaded_file is not None:
    # Audio playback
    st.markdown("### 🔊 Audio Playback")
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # Run classification
    st.markdown("### 🔬 Running Classification Pipeline...")
    progress = st.progress(0, text="Loading audio...")

    start_time = time.time()

    # Save uploaded file to a temporary buffer for librosa
    audio_bytes = io.BytesIO(uploaded_file.getvalue())

    progress.progress(15, text="Preprocessing audio (16kHz, Butterworth LPF)...")

    try:
        pred_class, class_name, confidence, all_probs, mswst, gamma, y_raw = \
            classify_audio(audio_bytes, model)
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
        st.stop()

    elapsed = time.time() - start_time
    progress.progress(100, text=f"Done in {elapsed:.2f}s!")
    time.sleep(0.5)
    progress.empty()

    # ---- Results Section ----
    st.markdown("---")
    st.markdown("## 🎯 Classification Result")

    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Predicted Class</div>
            <div class="prediction-class">{class_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Confidence</div>
            <div class="prediction-confidence">{confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Inference Time</div>
            <div class="prediction-class">{elapsed:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Confidence Bar Chart ----
    st.markdown("### 📊 Class Probabilities")

    fig_bar, ax_bar = plt.subplots(figsize=(10, 3))
    fig_bar.patch.set_facecolor('#0e1117')
    ax_bar.set_facecolor('#0e1117')

    classes = list(all_probs.keys())
    probs_list = list(all_probs.values())
    colors = ['#00d4ff' if p == max(probs_list) else '#4a5568' for p in probs_list]

    bars = ax_bar.barh(classes, probs_list, color=colors, height=0.6, edgecolor='none')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probability", color='#a0a0c0', fontsize=10)
    ax_bar.tick_params(colors='#a0a0c0', labelsize=9)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_color('#2d3748')
    ax_bar.spines['left'].set_color('#2d3748')

    for bar, prob in zip(bars, probs_list):
        ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', color='#e2e8f0', fontsize=10, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

    # ---- Visualizations ----
    st.markdown("---")
    st.markdown("## 📈 Signal & Feature Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    # Waveform
    with viz_col1:
        st.markdown("#### Waveform (16 kHz)")
        fig_wave, ax_wave = plt.subplots(figsize=(7, 3))
        fig_wave.patch.set_facecolor('#0e1117')
        ax_wave.set_facecolor('#0e1117')

        t = np.arange(len(y_raw)) / SR
        ax_wave.plot(t, y_raw, color='#00d4ff', linewidth=0.4, alpha=0.8)
        ax_wave.set_xlabel("Time (s)", color='#a0a0c0', fontsize=9)
        ax_wave.set_ylabel("Amplitude", color='#a0a0c0', fontsize=9)
        ax_wave.tick_params(colors='#a0a0c0', labelsize=8)
        ax_wave.spines['top'].set_visible(False)
        ax_wave.spines['right'].set_visible(False)
        ax_wave.spines['bottom'].set_color('#2d3748')
        ax_wave.spines['left'].set_color('#2d3748')
        plt.tight_layout()
        st.pyplot(fig_wave)
        plt.close()

    # Gammatone Spectrogram
    with viz_col2:
        st.markdown("#### Gammatone Filterbank (64 × 128)")
        fig_gamma, ax_gamma = plt.subplots(figsize=(7, 3))
        fig_gamma.patch.set_facecolor('#0e1117')
        ax_gamma.set_facecolor('#0e1117')

        im = ax_gamma.imshow(gamma, aspect='auto', origin='lower', cmap='inferno',
                             interpolation='bilinear')
        ax_gamma.set_xlabel("Time Bins", color='#a0a0c0', fontsize=9)
        ax_gamma.set_ylabel("ERB Channels", color='#a0a0c0', fontsize=9)
        ax_gamma.tick_params(colors='#a0a0c0', labelsize=8)
        plt.colorbar(im, ax=ax_gamma, shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig_gamma)
        plt.close()

    # MS-WST Heatmap
    st.markdown("#### Multi-Scale Wavelet Scattering Transform (4 × 128)")
    fig_wst, ax_wst = plt.subplots(figsize=(14, 2.5))
    fig_wst.patch.set_facecolor('#0e1117')
    ax_wst.set_facecolor('#0e1117')

    im2 = ax_wst.imshow(mswst, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax_wst.set_xlabel("Time Bins", color='#a0a0c0', fontsize=9)
    ax_wst.set_ylabel("Config", color='#a0a0c0', fontsize=9)
    ax_wst.set_yticks([0, 1, 2, 3])
    ax_wst.set_yticklabels(["J=4,Q=8", "J=6,Q=12", "J=7,Q=12", "J=7,Q=16"], fontsize=8)
    ax_wst.tick_params(colors='#a0a0c0', labelsize=8)
    plt.colorbar(im2, ax=ax_wst, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig_wst)
    plt.close()

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #a0a0c0;">
        <p style="font-size: 4rem; margin-bottom: 1rem;">🎤</p>
        <p style="font-size: 1.2rem; font-weight: 500;">Upload an audio file to get started</p>
        <p style="font-size: 0.9rem;">The model will classify the audio into one of four drone categories</p>
    </div>
    """, unsafe_allow_html=True)
