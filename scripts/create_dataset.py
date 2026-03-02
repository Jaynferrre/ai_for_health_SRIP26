"""
create_dataset.py — Sleep Signal Preprocessing & Dataset Creation
"""

import argparse
import os
import sys
import glob
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Breathing frequency band (Hz) — 10–24 BrPM
BREATH_LO_HZ = 0.17   # 10 BrPM
BREATH_HI_HZ = 0.40   # 24 BrPM

# Butterworth filter order — 4th order gives a good roll-off without ringing
FILTER_ORDER = 4

# Windowing parameters
WINDOW_DURATION_S = 30    # seconds
OVERLAP_FRACTION  = 0.50  # 50% overlap → step = 15 s
STEP_S = WINDOW_DURATION_S * (1 - OVERLAP_FRACTION)   # 15 s

# Labelling threshold — a window is assigned an event label only if the
# overlap between the window and the event exceeds this fraction of the
# window duration.
OVERLAP_THRESHOLD = 0.50  # 50%

# Target resampling rates after filtering (uniform across participants)
TARGET_FS_NASAL  = 32   # Hz  — same as raw; kept for explicitness
TARGET_FS_THORAC = 32   # Hz
TARGET_FS_SPO2   =  4   # Hz

# Events that carry clinical labels (all others → Normal if insufficient overlap)
CLINICAL_EVENTS = {
    "Hypopnea",
    "Obstructive Apnea",
    "Mixed Apnea",
    "Central Apnea",
}

# Header keys that identify the metadata section of signal files
SIGNAL_HEADER_KEYS = {"Signal ID:", "Start Time:", "Unit:", "Signal Type:"}


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_participants(in_dir: str) -> list[str]:
    candidates = sorted([
        d for d in glob.glob(os.path.join(in_dir, "*"))
        if os.path.isdir(d)
    ])
    if not candidates:
        sys.exit(f"ERROR: No sub-folders found in '{in_dir}'.")
    return candidates


def discover_files(folder: str) -> dict:
    txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    def find(include_kws, exclude_kws=None):
        ex = exclude_kws or []
        matches = [
            f for f in txt_files
            if all(k.lower() in f.lower() for k in include_kws)
            and not any(k.lower() in f.lower() for k in ex)
        ]
        if not matches:
            raise FileNotFoundError(
                f"No file in '{folder}' matching {include_kws} "
                f"(excluding {ex}). Files present: {txt_files}"
            )
        return os.path.join(folder, sorted(matches)[0])

    return {
        "nasal":  find(["flow"],           ["event", "events"]),
        "thorac": find(["thorac"]),
        "spo2":   find(["spo2"]),
        "events": find(["flow", "event"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS  (same logic as the reviewed notebook, isolated here for clarity)
# ─────────────────────────────────────────────────────────────────────────────

def _to_datetime(ts_str: str) -> datetime:
    return datetime.strptime(ts_str.replace(",", "."), "%d.%m.%Y %H:%M:%S.%f")


def parse_continuous(file_path: str) -> tuple[pd.Series, dict]:
    header, rows, in_data = {}, [], False

    with open(file_path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "Data:":
                in_data = True
                continue
            if not in_data:
                if ":" in line:
                    k, v = line.split(":", 1)
                    header[k.strip()] = v.strip()
            else:
                parts = line.split(";")
                if len(parts) != 2:
                    continue
                try:
                    ts  = _to_datetime(parts[0].strip())
                    val = float(parts[1].strip())
                    rows.append((ts, val))
                except ValueError:
                    continue

    if not rows:
        raise ValueError(f"No data rows parsed from: {file_path}")

    idx    = pd.DatetimeIndex([r[0] for r in rows])
    series = pd.Series([r[1] for r in rows], index=idx,
                       name=header.get("Signal Type", "Signal"))
    return series, header


def parse_events(file_path: str) -> pd.DataFrame:
    in_data, rows = False, []

    with open(file_path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # Switch to data mode once all header lines are consumed
            if not in_data and not any(
                line.startswith(h) for h in SIGNAL_HEADER_KEYS
            ):
                in_data = True
            if not in_data:
                continue

            parts = line.split(";")
            if len(parts) != 4:
                continue

            tr, dur_str, etype, stage = [p.strip() for p in parts]
            if "-" not in tr:
                continue

            start_str, end_str = tr.split("-", 1)
            try:
                start_dt = _to_datetime(start_str.strip())
                # End-time carries only HH:MM:SS,mmm — prepend date from start
                end_full  = start_dt.strftime("%d.%m.%Y") + " " + end_str.strip()
                end_dt    = _to_datetime(end_full)
                rows.append({
                    "Start":    start_dt,
                    "End":      end_dt,
                    "Duration": float(dur_str),
                    "Type":     etype,
                    "Stage":    stage,
                })
            except ValueError:
                continue

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Start", "End", "Duration", "Type", "Stage"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def design_bandpass(lo: float, hi: float, fs: float,
                    order: int = FILTER_ORDER):
    nyq = fs / 2.0
    lo_norm = lo / nyq
    hi_norm = hi / nyq

    # Clamp to valid Nyquist range with a small margin
    lo_norm = max(lo_norm, 1e-4)
    hi_norm = min(hi_norm, 1 - 1e-4)

    sos = butter(order, [lo_norm, hi_norm], btype="bandpass", output="sos")
    return sos


def apply_bandpass(signal_values: np.ndarray, fs: float,
                   lo: float = BREATH_LO_HZ,
                   hi: float = BREATH_HI_HZ) -> np.ndarray:
    if fs <= 0:
        return signal_values

    # SpO2 at 4 Hz: Nyquist = 2 Hz. The upper breathing edge (0.4 Hz) is
    # well within that, but the lower edge (0.17 Hz) needs enough samples
    # to resolve. Check minimum length.
    min_len = int(3 * fs / lo)   # at least 3 cycles of the lowest frequency
    if len(signal_values) < min_len:
        return signal_values   # too short to filter safely

    sos      = design_bandpass(lo, hi, fs)
    filtered = sosfiltfilt(sos, signal_values)
    return filtered


def resample_signal(series: pd.Series, source_fs: float,
                    target_fs: float) -> pd.Series:
    if source_fs == target_fs or len(series) == 0:
        return series

    # Compute up/down integers via GCD for exact rational resampling
    src = int(round(source_fs))
    tgt = int(round(target_fs))
    g   = gcd(src, tgt)
    up, down = tgt // g, src // g

    resampled_vals = resample_poly(series.values.astype(float), up, down)

    # Rebuild timestamp index at the new rate
    n_out    = len(resampled_vals)
    dt_ns    = int(1e9 / target_fs)           # nanoseconds per sample
    start_ns = series.index[0].value
    new_idx  = pd.DatetimeIndex(
        pd.array(
            [start_ns + i * dt_ns for i in range(n_out)],
            dtype="datetime64[ns]"
        )
    )

    return pd.Series(resampled_vals, index=new_idx, name=series.name)


# ─────────────────────────────────────────────────────────────────────────────
# LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def label_window(win_start: pd.Timestamp, win_end: pd.Timestamp,
                 events: pd.DataFrame,
                 threshold: float = OVERLAP_THRESHOLD,
                 clinical_only: bool = True) -> str:
    if events.empty:
        return "Normal"

    win_dur = (win_end - win_start).total_seconds()
    if win_dur <= 0:
        return "Normal"

    best_label   = "Normal"
    best_overlap = 0.0

    for _, ev in events.iterrows():
        if clinical_only and ev["Type"] not in CLINICAL_EVENTS:
            continue

        # Compute intersection
        overlap_start = max(win_start, ev["Start"])
        overlap_end   = min(win_end,   ev["End"])
        overlap_s     = (overlap_end - overlap_start).total_seconds()

        if overlap_s <= 0:
            continue

        frac = overlap_s / win_dur
        if frac > threshold and frac > best_overlap:
            best_overlap = frac
            best_label   = ev["Type"]

    return best_label


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(nasal_f:  pd.Series,
                 thorac_f: pd.Series,
                 spo2_f:   pd.Series,
                 events:   pd.DataFrame,
                 pid:      str,
                 nasal_fs:  float = TARGET_FS_NASAL,
                 thorac_fs: float = TARGET_FS_THORAC,
                 spo2_fs:   float = TARGET_FS_SPO2,
                 window_s:  float = WINDOW_DURATION_S,
                 step_s:    float = STEP_S) -> pd.DataFrame:
    # Expected number of samples per window for each signal
    n_nasal  = int(round(window_s * nasal_fs))
    n_thorac = int(round(window_s * thorac_fs))
    n_spo2   = int(round(window_s * spo2_fs))

    # Common recording span
    rec_start = max(nasal_f.index[0], thorac_f.index[0], spo2_f.index[0])
    rec_end   = min(nasal_f.index[-1], thorac_f.index[-1], spo2_f.index[-1])

    step_td   = pd.Timedelta(seconds=step_s)
    window_td = pd.Timedelta(seconds=window_s)

    rows      = []
    win_idx   = 0
    win_start = rec_start

    # Pre-convert events timestamps to pd.Timestamp for fast comparison
    if not events.empty:
        ev_copy = events.copy()
        ev_copy["Start"] = pd.to_datetime(ev_copy["Start"])
        ev_copy["End"]   = pd.to_datetime(ev_copy["End"])
    else:
        ev_copy = events

    while win_start + window_td <= rec_end:
        win_end = win_start + window_td

        # ── Extract raw window samples ──────────────────────────────────────
        def extract(sig: pd.Series, n_expected: int) -> np.ndarray | None:
            """Slice signal to window and return exactly n_expected samples."""
            mask   = (sig.index >= win_start) & (sig.index < win_end)
            chunk  = sig[mask].values
            if len(chunk) == 0:
                return None
            # Pad or trim to exact expected length
            if len(chunk) < n_expected:
                chunk = np.pad(chunk, (0, n_expected - len(chunk)),
                               mode="edge")
            elif len(chunk) > n_expected:
                chunk = chunk[:n_expected]
            return chunk

        n_vals  = extract(nasal_f,  n_nasal)
        t_vals  = extract(thorac_f, n_thorac)
        s_vals  = extract(spo2_f,   n_spo2)

        if n_vals is None or t_vals is None or s_vals is None:
            win_start += step_td
            win_idx   += 1
            continue

        # ── Label ─────────────────────────────────────────────────────────
        label = label_window(win_start, win_end, ev_copy)

        # ── Build row dict ─────────────────────────────────────────────────
        row = {
            "patient_id": pid,
            "window_idx": win_idx,
            "start_time": win_start,
            "end_time":   win_end,
        }
        for i, v in enumerate(n_vals):
            row[f"nasal_{i}"]  = v
        for i, v in enumerate(t_vals):
            row[f"thorac_{i}"] = v
        for i, v in enumerate(s_vals):
            row[f"spo2_{i}"]   = v
        row["label"] = label

        rows.append(row)
        win_start += step_td
        win_idx   += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PER-PARTICIPANT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_participant(folder: str, pid: str, log_lines: list) -> pd.DataFrame:
    log = lambda msg: (print(f"  {msg}"), log_lines.append(f"  {msg}"))

    log(f"{'─'*55}")
    log(f"Participant: {pid}   Folder: {folder}")

    # ── Discover files ──────────────────────────────────────────────────
    try:
        paths = discover_files(folder)
    except FileNotFoundError as e:
        log(f"SKIP — file discovery failed: {e}")
        return pd.DataFrame()

    # ── Parse ────────────────────────────────────────────────────────────
    try:
        nasal,  nasal_hdr  = parse_continuous(paths["nasal"])
        thorac, thorac_hdr = parse_continuous(paths["thorac"])
        spo2,   spo2_hdr   = parse_continuous(paths["spo2"])
        events             = parse_events(paths["events"])
    except Exception as e:
        log(f"SKIP — parse error: {e}")
        return pd.DataFrame()

    # Actual sampling rates from headers (fall back to expected defaults)
    fs_nasal  = float(nasal_hdr.get("Sample Rate",  TARGET_FS_NASAL))
    fs_thorac = float(thorac_hdr.get("Sample Rate", TARGET_FS_THORAC))
    fs_spo2   = float(spo2_hdr.get("Sample Rate",  TARGET_FS_SPO2))

    log(f"Raw  — Nasal: {len(nasal):>8,} @ {fs_nasal} Hz | "
        f"Thorac: {len(thorac):>8,} @ {fs_thorac} Hz | "
        f"SpO2: {len(spo2):>7,} @ {fs_spo2} Hz")
    log(f"Events parsed: {len(events)}")

    # ── Filter ───────────────────────────────────────────────────────────
    #
    # Bandpass filter design notes
    # ----------------------------
    # Nasal & Thoracic (32 Hz):
    #   Nyquist = 16 Hz.  Passband 0.17–0.40 Hz eliminates:
    #     - DC drift and very slow baseline wander (< 0.17 Hz)
    #     - Cardiac interference and motion artefacts (> 0.4 Hz)
    #   The filter is applied with sosfiltfilt (zero-phase) to avoid
    #   shifting breathing peaks relative to annotated event timestamps.
    #
    # SpO2 (4 Hz):
    #   Nyquist = 2 Hz.  The breathing band (0.17–0.40 Hz) is well within
    #   this range.  SpO2 desaturation caused by apnea typically lags the
    #   airflow event by 10–30 s; bandpass filtering preserves the slow
    #   desaturation kinetics while removing high-frequency sensor noise.

    log("Filtering signals to breathing band "
        f"[{BREATH_LO_HZ:.2f}–{BREATH_HI_HZ:.2f} Hz] ...")

    nasal_f  = pd.Series(
        apply_bandpass(nasal.values,  fs_nasal,  BREATH_LO_HZ, BREATH_HI_HZ),
        index=nasal.index,  name="Nasal Airflow Filtered"
    )
    thorac_f = pd.Series(
        apply_bandpass(thorac.values, fs_thorac, BREATH_LO_HZ, BREATH_HI_HZ),
        index=thorac.index, name="Thoracic Movement Filtered"
    )
    spo2_f   = pd.Series(
        apply_bandpass(spo2.values,   fs_spo2,   BREATH_LO_HZ, BREATH_HI_HZ),
        index=spo2.index,   name="SpO2 Filtered"
    )

    # ── Resample to target rates (if actual rate differs) ─────────────────
    if fs_nasal != TARGET_FS_NASAL:
        log(f"Resampling Nasal {fs_nasal}→{TARGET_FS_NASAL} Hz ...")
        nasal_f = resample_signal(nasal_f, fs_nasal, TARGET_FS_NASAL)
    if fs_thorac != TARGET_FS_THORAC:
        log(f"Resampling Thorac {fs_thorac}→{TARGET_FS_THORAC} Hz ...")
        thorac_f = resample_signal(thorac_f, fs_thorac, TARGET_FS_THORAC)
    if fs_spo2 != TARGET_FS_SPO2:
        log(f"Resampling SpO2 {fs_spo2}→{TARGET_FS_SPO2} Hz ...")
        spo2_f = resample_signal(spo2_f, fs_spo2, TARGET_FS_SPO2)

    # ── Window & label ────────────────────────────────────────────────────
    log(f"Windowing: {WINDOW_DURATION_S}s windows, "
        f"{int(OVERLAP_FRACTION*100)}% overlap (step={STEP_S}s) ...")

    windows_df = make_windows(
        nasal_f, thorac_f, spo2_f, events, pid,
        nasal_fs  = TARGET_FS_NASAL,
        thorac_fs = TARGET_FS_THORAC,
        spo2_fs   = TARGET_FS_SPO2,
    )

    if windows_df.empty:
        log("WARNING: No windows generated — check signal alignment.")
        return windows_df

    # ── Summary stats ────────────────────────────────────────────────────
    label_counts = windows_df["label"].value_counts()
    log(f"Windows generated: {len(windows_df)}")
    for lbl, cnt in label_counts.items():
        pct = 100 * cnt / len(windows_df)
        log(f"  {lbl:<25} {cnt:>5}  ({pct:5.1f}%)")

    return windows_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess sleep signals into labelled windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-in_dir",
        required=True,
        metavar="IN_DIR",
        help='Directory containing participant sub-folders, e.g. "Data"',
    )
    parser.add_argument(
        "-out_dir",
        required=True,
        metavar="OUT_DIR",
        help='Output directory for dataset files, e.g. "Dataset"',
    )
    args = parser.parse_args()

    in_dir  = args.in_dir.rstrip("/\\")
    out_dir = args.out_dir.rstrip("/\\")

    if not os.path.isdir(in_dir):
        sys.exit(f"ERROR: Input directory not found: '{in_dir}'")

    os.makedirs(out_dir, exist_ok=True)

    # ── Header ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("  create_dataset.py — Sleep Signal Dataset Builder")
    print("=" * 65)
    print(f"  Input  directory : {in_dir}")
    print(f"  Output directory : {out_dir}")
    print(f"  Bandpass         : {BREATH_LO_HZ} – {BREATH_HI_HZ} Hz "
          f"  ({BREATH_LO_HZ*60:.0f}–{BREATH_HI_HZ*60:.0f} BrPM)")
    print(f"  Window           : {WINDOW_DURATION_S}s, "
          f"{int(OVERLAP_FRACTION*100)}% overlap (step {STEP_S}s)")
    print(f"  Label threshold  : overlap > {int(OVERLAP_THRESHOLD*100)}% "
          f"of window")
    print("=" * 65)

    log_lines = []
    log_lines.append("create_dataset.py  —  Processing Log")
    log_lines.append(f"In : {in_dir}   Out: {out_dir}")
    log_lines.append(
        f"Bandpass: {BREATH_LO_HZ}–{BREATH_HI_HZ} Hz | "
        f"Window: {WINDOW_DURATION_S}s | Overlap: {int(OVERLAP_FRACTION*100)}% | "
        f"Label threshold: {int(OVERLAP_THRESHOLD*100)}%"
    )

    # ── Process each participant ─────────────────────────────────────────
    participant_folders = discover_participants(in_dir)
    print(f"\nFound {len(participant_folders)} participant folder(s).\n")

    all_dfs = []

    for folder in participant_folders:
        pid         = os.path.basename(folder)
        windows_df  = process_participant(folder, pid, log_lines)

        if windows_df.empty:
            print(f"  [{pid}] No data — skipped.\n")
            continue

        # Save per-participant CSV
        per_pid_path = os.path.join(out_dir, f"{pid}_windows.csv")
        windows_df.to_csv(per_pid_path, index=False)
        print(f"  [{pid}] Saved → {per_pid_path}  "
              f"({len(windows_df)} windows)\n")

        all_dfs.append(windows_df)

    if not all_dfs:
        sys.exit("ERROR: No data was successfully processed. Check input directory.")

    # ── Combine and save ─────────────────────────────────────────────────
    combined_df = pd.concat(all_dfs, ignore_index=True)

    csv_path = os.path.join(out_dir, "all_participants.csv")
    pkl_path = os.path.join(out_dir, "all_participants.pkl")

    combined_df.to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump(combined_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Summary report ───────────────────────────────────────────────────
    print("=" * 65)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total windows      : {len(combined_df):>8,}")
    print(f"  Participants       : {combined_df['patient_id'].nunique()}")
    n_nasal_cols  = sum(1 for c in combined_df.columns if c.startswith("nasal_"))
    n_thorac_cols = sum(1 for c in combined_df.columns if c.startswith("thorac_"))
    n_spo2_cols   = sum(1 for c in combined_df.columns if c.startswith("spo2_"))
    print(f"  Nasal samples/win  : {n_nasal_cols}  "
          f"({TARGET_FS_NASAL} Hz × {WINDOW_DURATION_S}s)")
    print(f"  Thorac samples/win : {n_thorac_cols}  "
          f"({TARGET_FS_THORAC} Hz × {WINDOW_DURATION_S}s)")
    print(f"  SpO2 samples/win   : {n_spo2_cols}  "
          f"({TARGET_FS_SPO2} Hz × {WINDOW_DURATION_S}s)")
    print(f"  Total features/win : {n_nasal_cols + n_thorac_cols + n_spo2_cols}")
    print()
    print("  Label distribution:")
    label_dist = combined_df["label"].value_counts()
    for lbl, cnt in label_dist.items():
        pct = 100 * cnt / len(combined_df)
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<25} {cnt:>6}  ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Output files:")
    print(f"    {csv_path}")
    print(f"    {pkl_path}")

    # Per-participant label breakdown
    log_lines.append("\n── COMBINED SUMMARY ──")
    log_lines.append(f"Total windows: {len(combined_df)}")
    for pid in combined_df["patient_id"].unique():
        sub   = combined_df[combined_df["patient_id"] == pid]
        lc    = sub["label"].value_counts()
        ahi_n = lc.get("Hypopnea", 0) + lc.get("Obstructive Apnea", 0) + \
                lc.get("Mixed Apnea", 0) + lc.get("Central Apnea", 0)
        log_lines.append(
            f"  {pid}: {len(sub)} windows | "
            + " | ".join(f"{k}={v}" for k, v in lc.items())
        )

    # Save log
    log_path = os.path.join(out_dir, "dataset_summary.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"    {log_path}")
    print()
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    main()
