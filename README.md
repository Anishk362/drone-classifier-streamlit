# 🛸 Acoustic Drone Detection & Classification v2

Multi-label drone classifier using **Multi-Scale Wavelet Scattering Transform (MS-WST)** and **Gammatone spectrograms** with cross-attention fusion.

## Features
- **Multi-label detection** — identifies 0, 1, 2, or 3 drone types simultaneously
- **3 drone classes**: DJI Mavic Mini, DJI Mavic 3 Cine, Agricultural Drone
- Dual-branch architecture with Scaled Dot-product Cross-Attention (SDCAM)
- Real-time audio classification via Streamlit web interface
- Interactive visualizations (waveform, MS-WST, Gammatone spectrogram)

## Architecture
1. **MS-WST Branch**: 4-scale wavelet scattering (Kymatio) → Conv1D decoder
2. **Gammatone Branch**: 64-channel filterbank spectrogram → Conv1D decoder  
3. **Fusion**: Cross-attention between branches → statistical pooling → MLP classifier
4. **Output**: Sigmoid activation for independent per-class detection

## Usage
Upload a WAV/MP3/FLAC/OGG audio file and the model will classify which drone(s) are present.

## Tech Stack
- PyTorch, Kymatio, Gammatone, Librosa
- Streamlit for the web interface
