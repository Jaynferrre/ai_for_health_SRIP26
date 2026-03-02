# Sleep Breathing Irregularity Detection
### AI for Health — SRIP 2026

A signal processing and machine learning pipeline for detecting breathing irregularities (apnea and hypopnea) during sleep, built on overnight polysomnography data from 5 participants.

---

## Repository Structure

```
ai_for_health_SRIP26/
│
├── Data/                          # Raw participant signal files
│   ├── AP01/
│   │   ├── Flow - 30-05-2024.txt          # Nasal Airflow (32 Hz)
│   │   ├── Thorac - 30-05-2024.txt        # Thoracic Movement (32 Hz)
│   │   ├── SPO2 - 30-05-2024.txt          # Oxygen Saturation (4 Hz)
│   │   ├── Flow Events - 30-05-2024.txt   # Breathing event annotations
│   │   └── Sleep profile - 30-05-2024.txt # Sleep stage epochs
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
│
├── Visualizations/                # Generated PDF reports
│   ├── AP01_sleep_report.pdf
│   ├── AP02_sleep_report.pdf
│   └── ...
│
├── Dataset/                       # Processed windowed datasets
│   ├── breathing_dataset.csv      # Windows labelled by breathing events
│   └── sleep_stage_dataset.csv    # Windows labelled by sleep stage
│
├── models/
│   ├── cnn_model.py               # 1D CNN classifier
│   └── conv_lstm_model.py         # ConvLSTM classifier
│
├── scripts/
│   ├── vis.py                     # Signal visualisation → PDF
│   ├── create_dataset.py          # Filtering, windowing, labelling
│   └── train.ipynb             # Model training & evaluation
│
├── README.md
└── report.pdf
```

---

## Dataset

Five participants (AP01–AP05) each contributed one 8-hour overnight sleep recording. Three physiological signals were recorded simultaneously:

| Signal | Sampling Rate | Description |
|---|---|---|
| Nasal Airflow | 32 Hz | Airflow measured at the nostrils via thermistor |
| Thoracic Movement | 32 Hz | Chest wall expansion via respiratory inductance plethysmography |
| SpO₂ | 4 Hz | Peripheral oxygen saturation via pulse oximetry |

Each recording also includes:
- **Flow Events file** — annotated breathing irregularities with start time, end time, duration, event type (Hypopnea, Obstructive Apnea, Mixed Apnea), and sleep stage
- **Sleep Profile file** — sleep stage labels in 30-second epochs (Wake, N1, N2, N3, REM, Movement)

---

## Setup

```bash
git clone https://github.com/<your-username>/ai_for_health_SRIP26.git
cd ai_for_health_SRIP26
pip install -r requirements.txt
```

**requirements**
```
numpy
pandas
scipy
matplotlib
scikit-learn
torch
```

---

## Pipeline

### Step 1 — Visualisation

Generate a multi-page PDF report for one participant showing all signals, event overlays, and a sleep hypnogram.

```bash
python scripts/vis.py -name "Data/AP01"
```

Output: `Visualizations/AP01_sleep_report.pdf`

The PDF contains:
- **Cover page** — recording metadata, AHI summary, event breakdown, sleep stage pie chart
- **Full 8-hour overview** — all 3 signals stacked with breathing events shaded and hypnogram strip
- **2-hour zoom panels** — signal detail with event type labels annotated directly on the trace
- **SpO₂ detail page** — desaturation histogram and cumulative burden curve
- **Event catalogue** — timeline rug, duration distribution, events per sleep stage

---

### Step 2 — Dataset Creation

Filter signals to the breathing band, segment into overlapping windows, and label each window.

```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

Output: `Dataset/breathing_dataset.csv` and `Dataset/sleep_stage_dataset.csv`

**Filtering**
A 4th-order zero-phase Butterworth bandpass filter (0.17–0.40 Hz) is applied to all signals using `sosfiltfilt`. This retains the breathing frequency band (10–24 BrPM) while removing:
- Baseline wander and DC drift (< 0.17 Hz)
- Cardiac interference and motion artefacts (> 0.40 Hz)

Zero-phase filtering (`sosfiltfilt`) is used to avoid temporal distortion of breathing events relative to annotations.

**Windowing**
- Window duration: **30 seconds**
- Overlap: **50%** (step = 15 seconds)
- Each window contains: 960 Nasal samples + 960 Thoracic samples + 120 SpO₂ samples

**Labelling**
| File | Label logic |
|---|---|
| `breathing_dataset.csv` | `label` — if a window overlaps > 50% of its duration with a breathing event → that event's type; otherwise `Normal` |
| `sleep_stage_dataset.csv` | `sleep_stage` — modal sleep stage epoch within the window; `breathing_label` also included |

---

### Step 3 — Model Training & Evaluation

```bash
python scripts/train_model.py -dataset "Dataset/breathing_dataset.csv"
```

**Model: 1D CNN**

```
Input  (batch, 2040, 1)    ← 960 nasal + 960 thorac + 120 SpO₂ samples
Conv1D(32, kernel=7) → BN → ReLU → MaxPool
Conv1D(64, kernel=5) → BN → ReLU → MaxPool
Conv1D(128, kernel=3) → BN → ReLU → GlobalAvgPool
Dense(64) → Dropout(0.5)
Dense(n_classes) → Softmax
```

**Evaluation: Leave-One-Participant-Out Cross-Validation (LOPO-CV)**

In each of 5 folds, the model is trained on 4 participants and tested on the held-out participant. This simulates real-world generalisation to unseen subjects — the clinically appropriate evaluation strategy for small-N health studies.

Metrics reported per fold and averaged:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- Confusion Matrix

---

## Clinical Context

Breathing irregularities are scored using the **Apnea-Hypopnea Index (AHI)** per AASM 2012 guidelines:

| AHI (events/hour) | Severity |
|---|---|
| < 5 | Normal |
| 5 – 14.9 | Mild OSA |
| 15 – 29.9 | Moderate OSA |
| ≥ 30 | Severe OSA |

**Important caveats:**
- This is a pilot study (N = 5). No classification result is statistically reliable at this sample size. All modelling is methodological demonstration.
- Sleep stage `A` (Arousal) appears in several participants but is absent from the documented `Events list` header. It is retained as a valid stage.
- SpO₂ desaturation caused by apnea typically lags the airflow event by 10–30 seconds. The zero-phase filter preserves this temporal relationship.

---

## Running in Google Colab

```python
# Clone repo and enter directory
!git clone https://github.com/<your-username>/ai_for_health_SRIP26.git
%cd ai_for_health_SRIP26

# Install dependencies
!pip install -r requirements.txt

# Run visualisation
!python scripts/vis.py -name "Data/AP01"

# Create dataset
!python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"

# Train and evaluate
!python scripts/train_model.py -dataset "Dataset/breathing_dataset.csv"

# Download outputs
from google.colab import files
files.download("Visualizations/AP01_sleep_report.pdf")
files.download("Dataset/breathing_dataset.csv")
files.download("Dataset/sleep_stage_dataset.csv")
```

---

## AI Tool Disclosure

This project was developed with assistance from **Claude (Anthropic)**. Per the assignment guidelines, this use is explicitly disclosed. The signal processing choices, clinical labelling strategy, and evaluation design were understood and verified by the submitting student.

---

## License

For academic use only — SRIP 2026 assessment submission.
