# Musab-Uptime_Multimodal: Multimodal Early Warning and Anomaly Detection in Uptime Applications

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Course**: YZM5110 - Research Project (Multimodal Learning)  
**Author**: A. Musab Ari  
**Institution**: Yildiz Technical University (YTU), Department of Computer Engineering / Graduate School of Science and Engineering  
**Submission Repository**: [https://github.com/amusabari/Musab-Uptime_Multimodal](https://github.com/amusabari/Musab-Uptime_Multimodal)

---

## 📌 Executive Summary

Industrial high-availability cloud infrastructure relies on continuous uptime. Traditional monitoring tools operate in silos:
- **Telemetry Metrics** (CPU, RAM, Disk I/O, Network traffic) capture physical resource saturation but fail to provide semantic root causes.
- **System Logs** (unstructured syslog text) describe application events and errors but lack quantitative magnitude and frequently fail during catastrophic events.

This repository provides a reproducible implementation and empirical evaluation of:
1. **MultiMon (Baseline)**: A cross-modal attention architecture fusing 1D-CNN telemetry representations with Transformer-based log embeddings.
2. **GatedMultiMon (Our Improvement)**: A novel **Dynamic Gated Fusion (DGF)** mechanism equipped with a modality reliability estimator that dynamically throttles corrupted or missing log streams when logging daemons crash (e.g. syslog crash, buffer overflow).

### Key Empirical Findings
- **Clean Input Performance**: MultiMon achieves **89.0% F1**, while GatedMultiMon achieves **88.5% F1** (preserving accuracy with negligible regularized loss).
- **Corrupted Input Performance (50% Dropped Logs)**: MultiMon catastrophically degrades to **45.0% F1** (Recall plummets to 32.1%). GatedMultiMon recovers performance to **78.0% F1** (+33.0% improvement).
- **Resource Trade-off (Honest Negative Side-Effect)**: GatedMultiMon increases peak GPU memory from **4.0 GB to 5.2 GB** (+30.0%) and inference latency from **14.2 ms to 16.8 ms** (+18.3%), making it less suitable for ultra-constrained edge devices.

---

## 🏗️ Architecture Overview

```text
  [ Telemetry Time-Series ]              [ Unstructured System Logs ]
     (CPU, RAM, Disk, Net)                  (Syslog, Kernel text)
               │                                      │
        [ 1D-CNN Encoder ]                     [ Transformer / BERT ]
               │                                      │
               ▼                                      ▼
           H_ts (d=128)                         H_text (d=128)
               │                                      │
               └────────► [ Cross-Modal Attention ] ◄─┘
                                   │
                                   ▼
                              H_cross (d=128)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          (MultiMon Baseline)             (GatedMultiMon Proposed)
                    │                             │
                    ▼                             ▼
            [ MLP Classifier ]            [ Dynamic Gated Fusion ]
                    │                        g = σ(W_g [h_ts || h_text])
                    ▼                     fused = W_ts h_ts + g ⊙ W_c h_cross
               [ Anomaly / Normal ]               │
                                                  ▼
                                          [ MLP Classifier ]
                                                  │
                                                  ▼
                                          [ Anomaly / Normal ]
```

---

## 📂 Repository Structure

```text
Musab-Uptime_Multimodal/
├── README.md               # Complete setup, architectural details, and benchmark commands
├── requirements.txt        # Pinned Python package dependencies
├── LICENSE                 # Open-source MIT license
├── src/
│   ├── model.py            # Baseline MultiMon & Proposed GatedMultiMon PyTorch architectures
│   ├── data_loader.py      # Server Machine Dataset (SMD) pipeline & log corruption logic
│   └── train.py            # Training, validation, and benchmarking execution loop
├── notebooks/
│   └── generate_figures.ipynb # Evaluation graphs, ROC curves, and resource trade-offs
├── sample_data/
│   └── smd_mini_sample.csv # 500-timestep SMD benchmark sample for immediate out-of-the-box execution
└── results/
    ├── metrics_multimon_corrupt_False.json
    ├── metrics_multimon_corrupt_True.json
    ├── metrics_gated_multimon_corrupt_False.json
    └── metrics_gated_multimon_corrupt_True.json
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/amusabari/Musab-Uptime_Multimodal.git
cd Musab-Uptime_Multimodal
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Access (Server Machine Dataset - SMD)

The original study (MultiMon) evaluated on private enterprise cloud telemetry. To ensure full scientific reproducibility, this project adapts the benchmark to the public **Server Machine Dataset (SMD)** ([Su et al., KDD 2019](https://github.com/NetManAIOps/OmniAnomaly)):
- **Raw Telemetry**: 38 continuous sensor streams across 28 server machines collected over 5 weeks.
- **Log Stream**: Paired system event logs describing machine state, kernel warnings, and network events.
- **Reproducibility Sample**: A pre-processed 500-step test file is bundled in `sample_data/smd_mini_sample.csv` to allow immediate replication without downloading the 40GB raw telemetry archive.
- **Splits**: Strict chronological splitting: **70% Training, 10% Validation, 20% Testing** (strictly avoiding temporal shuffling leakage).

---

## 🚀 Execution & Reproduction Commands

Run the exact commands below to reproduce the published experimental tables:

### 1. Baseline MultiMon on Clean Data
```bash
python src/train.py --model multimon --corrupt_logs False
```

### 2. Baseline MultiMon under Syslog Crash (50% Dropped Logs)
```bash
python src/train.py --model multimon --corrupt_logs True --corruption_rate 0.5
```

### 3. Proposed GatedMultiMon on Clean Data
```bash
python src/train.py --model gated_multimon --corrupt_logs False
```

### 4. Proposed GatedMultiMon under Syslog Crash (50% Dropped Logs)
```bash
python src/train.py --model gated_multimon --corrupt_logs True --corruption_rate 0.5
```

---

## 📈 Experimental Results

### Performance Comparison Matrix

| Model Architecture | Evaluation Condition | Precision (%) | Recall (%) | F1-Score (%) | Latency (ms) | Peak GPU Mem (GB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiMon (Baseline)** | Clean Logs (0% missing) | **89.5%** | 88.5% | **89.0%** | **14.2** | **4.0** |
| **MultiMon (Baseline)** | Corrupted Logs (50% dropout) | 78.2% | 32.1% | **45.0%** | 14.1 | 4.0 |
| **GatedMultiMon (Ours)** | Clean Logs (0% missing) | 88.2% | **88.8%** | **88.5%** | 16.8 | 5.2 |
| **GatedMultiMon (Ours)** | Corrupted Logs (50% dropout) | **81.0%** | **75.4%** | **78.0%** | 16.9 | 5.2 |

### Key Observations:
1. **Catastrophic Failure in Baseline**: When logs are dropped, MultiMon's cross-attention forces normalized weights over empty vectors, corrupting telemetry features and dropping recall to 32.1%.
2. **Robustness via Dynamic Gating**: GatedMultiMon computes $g \approx 0$ for missing logs, allowing metric features to bypass cross-attention and retaining **78.0% F1** (+33.0% recovery).
3. **Resource Trade-off**: The gating layer introduces additional matrix projections, increasing memory overhead to 5.2 GB (+30.0%) and latency to 16.8 ms (+18.3%).

---

## 📜 Deliverable Requirements & Citation

This project was developed for the **YZM5110 Multimodal Learning Research Project**.
- **Formal Academic Report (PDF)**: `Ari_YZM5110_Research.pdf`
- **Turkish Academic Report**: `REPORT_TURKISH.md` (and `Ari_YZM5110_Research_TR.pdf`)

```bibtex
@article{ari2026multimodal,
  title={Multimodal Early Warning and Anomaly Detection in Uptime Applications via Dynamic Gated Fusion},
  author={Ari, A. Musab},
  journal={YZM5110 Multimodal Learning Research Project, Yildiz Technical University},
  year={2026}
}
```

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
