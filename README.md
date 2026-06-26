# Drone Acoustic Classifier — Streamlit App

Real-time 4-class drone audio classification using a Dual-Feature Cross-Attention Deep Learning Pipeline.

## Classes
- **Class 0**: Background (No Drone)
- **Class 1**: DJI Mavic Mini
- **Class 2**: DJI Mavic 3 Cine
- **Class 3**: Agricultural Drone

## Architecture
- Multi-Scale Wavelet Scattering Transform (MS-WST) + Gammatone Filterbank
- Bidirectional Cross-Attention (SDCAM)
- 98.06% accuracy on IIT Jammu custom dataset

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author
Anish Kalra | 24MC3006 | IIT Jammu Research Internship 2025
