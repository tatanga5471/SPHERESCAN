# 🪐 Planet X & SPHEREx - NASA Mission Control

> **AI-Powered Astronomical Anomaly Detection & Multi-Epoch FITS Analysis Desktop Application**  
> Developed for the **NASA Space Apps Challenge**.

---

## 🚀 Overview

**Planet X & SPHEREx Mission Control** is a professional, native desktop application designed for astronomers and space enthusiasts to process FITS (Flexible Image Transport System) astronomical data, visualize multi-epoch observations, detect temporal changes, and uncover hidden cosmic anomalies using advanced Machine Learning models (**PyTorch ConvAutoencoders** and **Scikit-learn Isolation Forests**). 

Built with **PyQt6**, this application offers an offline-first, high-performance mission control interface without relying on web browser or localhost dependencies.

---

## ✨ Key Features

* **📁 Multi-Epoch FITS Management:** Load and compare astronomical observations across different epochs.
* **🎨 Advanced Visualization & Colormaps:** Dynamic percentiles-based contrast stretching (`vmin`/`vmax`) supporting multiple colormaps (`inferno`, `viridis`, `magma`, `plasma`, `gray`).
* **📊 Temporal Difference Analysis:** Automatic background-subtracted difference mapping with adjustable Sigma thresholds (`σ`) to isolate moving or changing sources.
* **🌲 Unsupervised Anomaly Detection:** Powered by **Isolation Forest** to flag structural outliers in image patches.
* **🤖 Deep Learning AI Scan:** Real-time **PyTorch Convolutional Autoencoder** reconstruction error analysis to detect rare spatial anomalies with live anomaly score ranking.
* **📈 Photometric Light Curve:** Interactive flux tracking across epochs for any selected pixel coordinate `(X, Y)`.
* **📥 CSV Data Export:** Instantly export detected candidates and AI anomaly report tables into standard CSV files for further scientific analysis.

---

## 🛠️ Technology Stack

* **GUI Framework:** PyQt6, Matplotlib (Embedded Qt Canvas)
* **Data Processing & Astronomy:** Astropy, NumPy, SciPy
* **Machine Learning & Deep Learning:** PyTorch, Scikit-learn
* **Packaging:** PyInstaller (Standalone `.exe` generation)

---

## 📂 Project Structure

```text
PlanetX_SPHEREx/
│
├── .venv/                      # Python Virtual Environment
├── models/                     # Saved PyTorch Autoencoder weights (.pth)
├── ai_results/                 # Local analysis reports and logs
├── data_loader.py              # FITS file reader & WCS coordinate parser
├── preprocessing.py            # Image difference & alignment modules
├── analysis.py                 # Statistical candidate change detection
├── ml_model.py                 # ConvAutoencoder & Isolation Forest models
├── main_gui.py                 # Main PyQt6 Mission Control interface
└── README.md                   # Project documentation
