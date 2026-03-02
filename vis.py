"""
vis.py — Sleep Signal Visualisation Tool
"""

import argparse
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & STYLE
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLOURS = {
    "Nasal Airflow":     "#2196F3",   # blue
    "Thoracic Movement": "#FF9800",   # orange
    "SpO2":              "#4CAF50",   # green
}

EVENT_COLOURS = {
    "Hypopnea":          "#FFC107",   # amber
    "Obstructive Apnea": "#F44336",   # red
    "Mixed Apnea":       "#9C27B0",   # purple
    "Central Apnea":     "#E91E63",   # pink
    "Body event":        "#9E9E9E",   # grey
}
DEFAULT_EVENT_COLOUR = "#607D8B"

STAGE_ORDER   = ["Wake", "REM", "N1", "N2", "N3", "N4", "Movement", "A"]
STAGE_COLOURS = {
    "Wake":     "#EF5350",
    "REM":      "#CE93D8",
    "N1":       "#90CAF9",
    "N2":       "#42A5F5",
    "N3":       "#1565C0",
    "N4":       "#0D47A1",
    "Movement": "#FFD54F",
    "A":        "#FF7043",   # Arousal — undocumented in some files but present
}

WAKE_STAGES = {"Wake", "Movement"}
APNEA_EVENTS  = {"Obstructive Apnea", "Mixed Apnea", "Central Apnea"}
HYPOPNEA_EVENTS = {"Hypopnea"}
EPOCH_S = 30          # sleep profile epoch duration in seconds

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.linewidth": 0.8,
    "axes.grid":      True,
    "grid.alpha":     0.25,
    "grid.linewidth": 0.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(folder: str) -> dict:
    """
    Locate signal files inside *folder* using keyword matching.
    File names vary across participants (different date separators, signal
    name variants), so we match by keyword rather than exact name.

    Returns a dict with keys:
        nasal, thorac, spo2, events, sleep
    Raises FileNotFoundError if any required file is missing.
    """
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    def find(keywords_include, keywords_exclude=None):
        kw_ex = keywords_exclude or []
        matches = [
            f for f in files
            if all(k.lower() in f.lower() for k in keywords_include)
            and not any(k.lower() in f.lower() for k in kw_ex)
        ]
        if not matches:
            raise FileNotFoundError(
                f"Cannot find file in '{folder}' matching {keywords_include} "
                f"(excluding {kw_ex}). Found: {files}"
            )
        return os.path.join(folder, sorted(matches)[0])

    return {
        "nasal":  find(["flow"], ["event", "events"]),
        "thorac": find(["thorac"]),
        "spo2":   find(["spo2"]),
        "events": find(["flow", "event"]),
        "sleep":  find(["sleep", "profile"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> datetime:
    """Parse 'DD.MM.YYYY HH:MM:SS.f' (comma already replaced with period)."""
    return datetime.strptime(ts_str, "%d.%m.%Y %H:%M:%S.%f")


def parse_continuous(file_path: str) -> tuple[pd.Series, dict]:
    """
    Parse a continuous-signal file (Nasal Airflow / Thoracic Movement / SpO2).

    Returns
    -------
    series      : pd.Series with DatetimeIndex, name = signal type from header
    header_info : dict of metadata
    """
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
                    ts  = _parse_ts(parts[0].strip().replace(",", "."))
                    val = float(parts[1].strip())
                    rows.append((ts, val))
                except ValueError:
                    continue

    if not rows:
        raise ValueError(f"No data parsed from {file_path}")

    idx = pd.DatetimeIndex([r[0] for r in rows])
    s   = pd.Series([r[1] for r in rows], index=idx,
                    name=header.get("Signal Type", "Signal"))
    return s, header


def parse_events(file_path: str) -> pd.DataFrame:
    """
    Parse a Flow Events annotation file.

    Each row: DD.MM.YYYY HH:MM:SS,mmm-HH:MM:SS,mmm; duration_s; EventType; Stage
    End-time contains only the time portion → date is inherited from start.
    """
    HEADER_KEYS = {"Signal ID:", "Start Time:", "Unit:", "Signal Type:"}
    in_data, rows = False, []

    with open(file_path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if not in_data and not any(line.startswith(h) for h in HEADER_KEYS):
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
                start_dt = _parse_ts(start_str.strip().replace(",", "."))
                end_full  = start_dt.strftime("%d.%m.%Y") + " " + end_str.strip()
                end_dt    = _parse_ts(end_full.replace(",", "."))
                rows.append({
                    "Start":    start_dt,
                    "End":      end_dt,
                    "Duration": float(dur_str),
                    "Type":     etype,
                    "Stage":    stage,
                })
            except ValueError:
                continue

    return pd.DataFrame(rows)


def parse_sleep(file_path: str) -> pd.Series:
    """
    Parse a Sleep Profile file (discrete 30-second epochs).
    Returns pd.Series with DatetimeIndex and sleep stage strings.
    """
    in_data, rows = False, []

    with open(file_path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if "Rate:" in line:
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(";")
            if len(parts) != 2:
                continue
            try:
                ts    = _parse_ts(parts[0].strip().replace(",", "."))
                stage = parts[1].strip()
                rows.append((ts, stage))
            except ValueError:
                continue

    if not rows:
        return pd.Series(dtype=str)
    idx = pd.DatetimeIndex([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name="Sleep Stage")


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(events: pd.DataFrame, sleep: pd.Series,
                    spo2: pd.Series) -> dict:
    """Compute AHI, TST, sleep efficiency, T90, event counts."""
    sleep_epochs = sleep[~sleep.isin(WAKE_STAGES)]
    tst_s  = len(sleep_epochs) * EPOCH_S
    tst_h  = tst_s / 3600
    tib_h  = len(sleep) * EPOCH_S / 3600

    n_ap   = events["Type"].isin(APNEA_EVENTS).sum()
    n_hyp  = events["Type"].isin(HYPOPNEA_EVENTS).sum()
    ahi    = (n_ap + n_hyp) / tst_h if tst_h > 0 else 0

    spo2_v = spo2.dropna()
    t90    = 100 * (spo2_v < 90).sum() / len(spo2_v) if len(spo2_v) else 0
    eff    = 100 * tst_h / tib_h if tib_h > 0 else 0

    def severity(a):
        if a < 5:   return "Normal (No OSA)"
        if a < 15:  return "Mild OSA"
        if a < 30:  return "Moderate OSA"
        return "Severe OSA"

    return {
        "TST_h": round(tst_h, 2),
        "TIB_h": round(tib_h, 2),
        "Sleep Efficiency (%)": round(eff, 1),
        "AHI (events/h)": round(ahi, 1),
        "Severity": severity(ahi),
        "N Apneas": int(n_ap),
        "N Hypopneas": int(n_hyp),
        "N Total Events": len(events),
        "SpO2 Mean (%)": round(float(spo2_v.mean()), 1) if len(spo2_v) else "N/A",
        "SpO2 Min (%)":  round(float(spo2_v.min()),  1) if len(spo2_v) else "N/A",
        "T90 (%)": round(t90, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_axis(ax, ylabel, colour):
    ax.set_ylabel(ylabel, color=colour, fontsize=8, labelpad=4)
    ax.tick_params(axis="y", labelcolor=colour)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)


def _shade_events(ax, events: pd.DataFrame, ymin=0, ymax=1):
    """Shade breathing event windows on axis *ax* (uses axis transform)."""
    for _, row in events.iterrows():
        ec = EVENT_COLOURS.get(row["Type"], DEFAULT_EVENT_COLOUR)
        ax.axvspan(row["Start"], row["End"], ymin=ymin, ymax=ymax,
                   color=ec, alpha=0.25, linewidth=0, zorder=2)


def _event_legend_handles():
    return [
        mpatches.Patch(color=c, alpha=0.6, label=k)
        for k, c in EVENT_COLOURS.items()
    ]


def _downsample(series: pd.Series, max_pts: int = 20_000) -> pd.Series:
    """Thin a high-frequency series for fast rendering without losing peaks."""
    if len(series) <= max_pts:
        return series
    step = len(series) // max_pts
    # Keep every step-th point AND local min/max within each window
    idx_keep = np.arange(0, len(series), step)
    return series.iloc[idx_keep]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

# --- Page 1 : Cover ---

def page_cover(pdf: PdfPages, pid: str, folder: str,
               nasal_hdr: dict, spo2_hdr: dict,
               events: pd.DataFrame, sleep: pd.Series,
               metrics: dict):

    fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
    fig.patch.set_facecolor("#F5F5F5")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Title bar
    fig.add_axes([0, 0.88, 1, 0.12]).set_axis_off()
    fig.text(0.5, 0.93, "Sleep Study — Signal Visualisation Report",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color="#1A237E")
    fig.text(0.5, 0.895, f"Participant: {pid}    |    Folder: {folder}",
             ha="center", va="center", fontsize=10, color="#37474F")

    # Recording metadata box
    rec_start = nasal_hdr.get("Start Time", "N/A")
    fs_nasal  = nasal_hdr.get("Sample Rate", "?")
    fs_spo2   = spo2_hdr.get("Sample Rate",  "?")
    meta_lines = [
        f"Recording Start : {rec_start}",
        f"Nasal/Thoracic Fs : {fs_nasal} Hz",
        f"SpO2 Fs : {fs_spo2} Hz",
        f"Sleep Epochs : {len(sleep)} x {EPOCH_S}s",
    ]
    fig.text(0.05, 0.83, "Recording Metadata",
             fontsize=11, fontweight="bold", color="#1565C0")
    for i, line in enumerate(meta_lines):
        fig.text(0.05, 0.79 - i * 0.035, line, fontsize=9, color="#212121")

    # Clinical metrics table
    fig.text(0.05, 0.63, "Clinical Summary (AASM Guidelines)",
             fontsize=11, fontweight="bold", color="#1565C0")

    col_labels = list(metrics.keys())
    col_vals   = [str(v) for v in metrics.values()]
    n = len(col_labels)
    col_w = 0.9 / n
    x0    = 0.05

    # Header row
    for j, (lbl, val) in enumerate(zip(col_labels, col_vals)):
        x = x0 + j * col_w
        fig.add_axes([x, 0.53, col_w - 0.005, 0.07]).set_axis_off()
        # draw via text
        fig.text(x + col_w / 2, 0.595, lbl,
                 ha="center", va="center", fontsize=6.5,
                 fontweight="bold", color="#FAFAFA",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#1565C0", edgecolor="none",
                           alpha=0.9))
        # severity gets colour-coded
        fc = "#FFFFFF"
        if lbl == "Severity":
            fc_map = {
                "Normal (No OSA)": "#C8E6C9",
                "Mild OSA":        "#FFF9C4",
                "Moderate OSA":    "#FFE0B2",
                "Severe OSA":      "#FFCDD2",
            }
            fc = fc_map.get(val, "#FFFFFF")

        fig.text(x + col_w / 2, 0.548, val,
                 ha="center", va="center", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=fc, edgecolor="#BDBDBD", linewidth=0.5))

    # Event type breakdown bar chart
    ev_counts = events["Type"].value_counts()
    if len(ev_counts):
        ax_bar = fig.add_axes([0.05, 0.25, 0.40, 0.22])
        colours_bar = [EVENT_COLOURS.get(t, DEFAULT_EVENT_COLOUR) for t in ev_counts.index]
        bars = ax_bar.barh(ev_counts.index, ev_counts.values,
                           color=colours_bar, edgecolor="white", height=0.6)
        ax_bar.set_xlabel("Count", fontsize=8)
        ax_bar.set_title("Breathing Event Breakdown", fontsize=9, fontweight="bold")
        ax_bar.tick_params(axis="y", labelsize=8)
        for bar, val in zip(bars, ev_counts.values):
            ax_bar.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=8)
        ax_bar.set_facecolor("#FAFAFA")

    # Sleep stage pie chart
    stage_counts = sleep.value_counts()
    if len(stage_counts):
        ax_pie = fig.add_axes([0.55, 0.22, 0.38, 0.28])
        pie_colours = [STAGE_COLOURS.get(s, "#BDBDBD") for s in stage_counts.index]
        wedges, texts, autotexts = ax_pie.pie(
            stage_counts.values,
            labels=stage_counts.index,
            colors=pie_colours,
            autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
            startangle=90,
            pctdistance=0.75,
            textprops={"fontsize": 7},
        )
        for at in autotexts:
            at.set_fontsize(6.5)
        ax_pie.set_title("Sleep Stage Distribution", fontsize=9,
                         fontweight="bold", pad=8)

    # Footer
    fig.text(0.5, 0.03,
             "AHI thresholds per AASM 2012 guidelines  |  "
             "Generated by vis.py  |  For research use only",
             ha="center", fontsize=7, color="#9E9E9E")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# --- Page 2 : Full 8-hour Overview ---

def page_overview(pdf: PdfPages, pid: str,
                  nasal: pd.Series, thorac: pd.Series, spo2: pd.Series,
                  events: pd.DataFrame, sleep: pd.Series):

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"{pid}  —  Full 8-Hour Sleep Study Overview",
                 fontsize=12, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(4, 1, hspace=0.08,
                           top=0.96, bottom=0.09, left=0.08, right=0.97,
                           height_ratios=[3, 3, 3, 1.5])

    rec_start = min(nasal.index[0], thorac.index[0], spo2.index[0])
    rec_end   = max(nasal.index[-1], thorac.index[-1], spo2.index[-1])

    # ── Nasal Airflow ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(_downsample(nasal).index, _downsample(nasal).values,
             color=SIGNAL_COLOURS["Nasal Airflow"], linewidth=0.5, alpha=0.85)
    _shade_events(ax1, events)
    _fmt_axis(ax1, "Nasal Airflow\n(a.u.)", SIGNAL_COLOURS["Nasal Airflow"])
    ax1.set_xlim(rec_start, rec_end)
    ax1.set_xticklabels([])
    ax1.set_title("Nasal Airflow", loc="left", fontsize=8, pad=2)

    # ── Thoracic Movement ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(_downsample(thorac).index, _downsample(thorac).values,
             color=SIGNAL_COLOURS["Thoracic Movement"], linewidth=0.5, alpha=0.85)
    _shade_events(ax2, events)
    _fmt_axis(ax2, "Thoracic Mvmt\n(a.u.)", SIGNAL_COLOURS["Thoracic Movement"])
    ax2.set_xticklabels([])
    ax2.set_title("Thoracic Movement", loc="left", fontsize=8, pad=2)

    # ── SpO2 ───────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(spo2.index, spo2.values,
             color=SIGNAL_COLOURS["SpO2"], linewidth=0.8, alpha=0.9)
    ax3.axhline(90, color="#E53935", linewidth=1.0, linestyle="--",
                label="SpO2 = 90%", zorder=5)
    ax3.fill_between(spo2.index, spo2.values, 90,
                     where=(spo2.values < 90),
                     color="#FFCDD2", alpha=0.5, label="Below 90%")
    _shade_events(ax3, events)
    _fmt_axis(ax3, "SpO2 (%)", SIGNAL_COLOURS["SpO2"])
    ax3.set_ylim(max(60, float(spo2.min()) - 3), 102)
    ax3.legend(fontsize=7, loc="lower right")
    ax3.set_xticklabels([])
    ax3.set_title("SpO2 (Oxygen Saturation)", loc="left", fontsize=8, pad=2)

    # ── Hypnogram ──────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    stage_y = {s: i for i, s in enumerate(reversed(STAGE_ORDER))}
    sp_reset = sleep.reset_index()
    sp_reset.columns = ["Timestamp", "Stage"]

    for _, epoch in sp_reset.iterrows():
        s = epoch["Stage"]
        y = stage_y.get(s, 0)
        c = STAGE_COLOURS.get(s, "#BDBDBD")
        ax4.barh(y, EPOCH_S / 3600, left=mdates.date2num(epoch["Timestamp"]),
                 height=0.85, color=c, linewidth=0, alpha=0.9)

    ax4.set_yticks(list(stage_y.values()))
    ax4.set_yticklabels([s for s in reversed(STAGE_ORDER)], fontsize=6.5)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax4.set_xlabel("Clock Time", fontsize=8)
    ax4.set_title("Sleep Hypnogram", loc="left", fontsize=8, pad=2)
    ax4.grid(False)

    # ── Event legend ────────────────────────────────────────────────────────
    handles = _event_legend_handles()
    fig.legend(handles=handles, title="Breathing Events",
               loc="lower center", ncol=len(handles),
               fontsize=7, title_fontsize=7,
               bbox_to_anchor=(0.5, 0.01),
               framealpha=0.8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# --- Pages 3-5 : 2-Hour Zoom Panels ---

def pages_zoom(pdf: PdfPages, pid: str,
               nasal: pd.Series, thorac: pd.Series, spo2: pd.Series,
               events: pd.DataFrame, sleep: pd.Series,
               window_hours: float = 2.0):

    rec_start = min(nasal.index[0], thorac.index[0], spo2.index[0])
    rec_end   = max(nasal.index[-1], thorac.index[-1], spo2.index[-1])

    total_h = (rec_end - rec_start).total_seconds() / 3600
    n_windows = int(np.ceil(total_h / window_hours))

    for w in range(n_windows):
        t0 = rec_start + pd.Timedelta(hours=w * window_hours)
        t1 = t0 + pd.Timedelta(hours=window_hours)
        t1 = min(t1, rec_end)

        # Slice all signals to window
        def slc(s, a, b):
            mask = (s.index >= a) & (s.index <= b)
            return s[mask]

        n_w = slc(nasal,  t0, t1)
        t_w = slc(thorac, t0, t1)
        s_w = slc(spo2,   t0, t1)
        ev_w = events[(events["Start"] >= t0) & (events["Start"] <= t1)]
        sp_w = sleep[(sleep.index >= t0) & (sleep.index <= t1)]

        fig = plt.figure(figsize=(11.69, 8.27))
        hour_label = f"{w*window_hours:.0f}h – {min((w+1)*window_hours, total_h):.1f}h"
        fig.suptitle(f"{pid}  —  Detail View: {hour_label}",
                     fontsize=11, fontweight="bold", y=0.99)

        gs = gridspec.GridSpec(4, 1, hspace=0.07,
                               top=0.95, bottom=0.09, left=0.08, right=0.97,
                               height_ratios=[3, 3, 3, 1.5])

        # Nasal
        ax1 = fig.add_subplot(gs[0])
        if len(n_w):
            ax1.plot(n_w.index, n_w.values,
                     color=SIGNAL_COLOURS["Nasal Airflow"], linewidth=0.6, alpha=0.9)
        _shade_events(ax1, ev_w)
        _fmt_axis(ax1, "Nasal Airflow\n(a.u.)", SIGNAL_COLOURS["Nasal Airflow"])
        ax1.set_xlim(t0, t1)
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 121, 15)))
        ax1.set_xticklabels([])
        ax1.set_title("Nasal Airflow", loc="left", fontsize=8)

        # Annotate events on nasal signal
        for _, ev in ev_w.iterrows():
            mid = ev["Start"] + (ev["End"] - ev["Start"]) / 2
            y_pos = ax1.get_ylim()[1] * 0.88
            label = ev["Type"].replace("Obstructive ", "Obstr.\n").replace("Apnea", "Apnea")
            ax1.text(mid, y_pos, label, ha="center", va="top",
                     fontsize=5.5, color="#B71C1C",
                     bbox=dict(facecolor="white", alpha=0.6,
                               edgecolor="none", pad=1))

        # Thoracic
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if len(t_w):
            ax2.plot(t_w.index, t_w.values,
                     color=SIGNAL_COLOURS["Thoracic Movement"], linewidth=0.6, alpha=0.9)
        _shade_events(ax2, ev_w)
        _fmt_axis(ax2, "Thoracic Mvmt\n(a.u.)", SIGNAL_COLOURS["Thoracic Movement"])
        ax2.set_xticklabels([])
        ax2.set_title("Thoracic Movement", loc="left", fontsize=8)

        # SpO2
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if len(s_w):
            ax3.plot(s_w.index, s_w.values,
                     color=SIGNAL_COLOURS["SpO2"], linewidth=1.0, alpha=0.95)
            ax3.fill_between(s_w.index, s_w.values, 90,
                             where=(s_w.values < 90),
                             color="#FFCDD2", alpha=0.55)
        ax3.axhline(90, color="#E53935", linewidth=1.0, linestyle="--", zorder=5)
        _shade_events(ax3, ev_w)
        _fmt_axis(ax3, "SpO2 (%)", SIGNAL_COLOURS["SpO2"])
        y_lo = max(60, float(s_w.min()) - 3) if len(s_w) else 85
        ax3.set_ylim(y_lo, 102)
        ax3.set_xticklabels([])
        ax3.set_title("SpO2", loc="left", fontsize=8)

        # Hypnogram strip
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        stage_y = {s: i for i, s in enumerate(reversed(STAGE_ORDER))}
        sp_r = sp_w.reset_index()
        if not sp_r.empty:
            sp_r.columns = ["Timestamp", "Stage"]
            for _, epoch in sp_r.iterrows():
                s = epoch["Stage"]
                y = stage_y.get(s, 0)
                c = STAGE_COLOURS.get(s, "#BDBDBD")
                ax4.barh(y, EPOCH_S / 3600 / (window_hours),
                         left=mdates.date2num(epoch["Timestamp"]),
                         height=0.85, color=c, linewidth=0, alpha=0.9)
        ax4.set_yticks(list(stage_y.values()))
        ax4.set_yticklabels([s for s in reversed(STAGE_ORDER)], fontsize=6)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 121, 15)))
        plt.setp(ax4.get_xticklabels(), rotation=30, ha="right", fontsize=7)
        ax4.set_xlabel("Clock Time", fontsize=8)
        ax4.set_title("Hypnogram", loc="left", fontsize=8)
        ax4.grid(False)

        # Event count annotation
        if len(ev_w):
            fig.text(0.97, 0.5,
                     f"{len(ev_w)} event(s)\nin window",
                     ha="right", va="center", fontsize=8,
                     color="#B71C1C", rotation=90)

        # Legend
        handles = _event_legend_handles()
        fig.legend(handles=handles, title="Events",
                   loc="lower center", ncol=len(handles),
                   fontsize=7, title_fontsize=7,
                   bbox_to_anchor=(0.5, 0.01), framealpha=0.8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# --- Page 6 : SpO2 Detail ---

def page_spo2_detail(pdf: PdfPages, pid: str,
                     spo2: pd.Series, events: pd.DataFrame):

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"{pid}  —  SpO2 Detail Analysis",
                 fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35,
                           top=0.93, bottom=0.09, left=0.08, right=0.97)

    # Full trace
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(spo2.index, spo2.values,
             color=SIGNAL_COLOURS["SpO2"], linewidth=0.7, alpha=0.9)
    ax1.axhline(90, color="#E53935", linewidth=1.2, linestyle="--", label="90%")
    ax1.axhline(85, color="#B71C1C", linewidth=0.8, linestyle=":", label="85%")
    ax1.fill_between(spo2.index, spo2.values, 90,
                     where=(spo2.values < 90),
                     color="#FFCDD2", alpha=0.55, label="Below 90%")
    _shade_events(ax1, events)
    ax1.set_ylabel("SpO2 (%)")
    ax1.set_xlabel("Clock Time")
    ax1.set_title("Full-Night SpO2 Trace", fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.legend(fontsize=8)
    ax1.set_ylim(max(60, float(spo2.min()) - 3), 102)

    # Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    vals = spo2.dropna().values
    ax2.hist(vals, bins=50, color=SIGNAL_COLOURS["SpO2"],
             edgecolor="white", linewidth=0.3, alpha=0.85)
    ax2.axvline(90, color="#E53935", linewidth=1.2, linestyle="--", label="90%")
    t90 = 100 * (vals < 90).sum() / len(vals)
    ax2.set_title(f"SpO2 Distribution  (T90 = {t90:.1f}%)", fontweight="bold")
    ax2.set_xlabel("SpO2 (%)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)

    # Cumulative desaturation burden
    ax3 = fig.add_subplot(gs[1, 1])
    thresholds = np.arange(80, 101, 1)
    pct_below  = [100 * (vals < t).sum() / len(vals) for t in thresholds]
    ax3.plot(thresholds, pct_below, color="#1565C0", linewidth=1.5)
    ax3.axvline(90, color="#E53935", linewidth=1.0, linestyle="--")
    ax3.fill_between(thresholds, pct_below, alpha=0.2, color="#1565C0")
    ax3.set_xlabel("SpO2 Threshold (%)")
    ax3.set_ylabel("% Recording Time Below Threshold")
    ax3.set_title("Cumulative Desaturation Burden", fontweight="bold")
    ax3.invert_xaxis()

    # Stat box
    stats_txt = (
        f"Mean  : {vals.mean():.1f}%\n"
        f"Median: {np.median(vals):.1f}%\n"
        f"Min   : {vals.min():.1f}%\n"
        f"T90   : {t90:.2f}%"
    )
    ax3.text(0.02, 0.98, stats_txt, transform=ax3.transAxes,
             va="top", fontsize=8, family="monospace",
             bbox=dict(facecolor="white", edgecolor="#BDBDBD",
                       boxstyle="round,pad=0.4"))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# --- Page 7 : Event Catalogue ---

def page_event_catalogue(pdf: PdfPages, pid: str, events: pd.DataFrame):

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(f"{pid}  —  Breathing Event Catalogue",
                 fontsize=12, fontweight="bold")

    if events.empty:
        fig.text(0.5, 0.5, "No breathing events recorded.",
                 ha="center", va="center", fontsize=14, color="#9E9E9E")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35,
                           top=0.92, bottom=0.09, left=0.08, right=0.97)

    # Timeline rug of events
    ax1 = fig.add_subplot(gs[0, :])
    for _, row in events.iterrows():
        ec = EVENT_COLOURS.get(row["Type"], DEFAULT_EVENT_COLOUR)
        ax1.barh(0, (row["End"] - row["Start"]).total_seconds() / 3600,
                 left=mdates.date2num(row["Start"]),
                 height=0.6, color=ec, alpha=0.8, linewidth=0)
    ax1.set_yticks([])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax1.set_title("Event Timeline (full night)", fontweight="bold")
    ax1.set_xlabel("Clock Time")
    handles = _event_legend_handles()
    ax1.legend(handles=handles, fontsize=7, loc="upper right")

    # Duration distribution
    ax2 = fig.add_subplot(gs[1, 0])
    for etype, grp in events.groupby("Type"):
        ax2.hist(grp["Duration"], bins=20, alpha=0.6, label=etype,
                 color=EVENT_COLOURS.get(etype, DEFAULT_EVENT_COLOUR),
                 edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Duration (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Event Duration Distribution", fontweight="bold")
    ax2.legend(fontsize=7)

    # Events per sleep stage
    ax3 = fig.add_subplot(gs[1, 1])
    stage_ev = events.groupby(["Stage", "Type"]).size().unstack(fill_value=0)
    colours_bar = [EVENT_COLOURS.get(c, DEFAULT_EVENT_COLOUR)
                   for c in stage_ev.columns]
    stage_ev.plot(kind="bar", stacked=True, ax=ax3,
                  color=colours_bar, edgecolor="white", linewidth=0.3)
    ax3.set_title("Events per Sleep Stage", fontweight="bold")
    ax3.set_xlabel("Sleep Stage")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7, loc="upper right")
    plt.setp(ax3.get_xticklabels(), rotation=0, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a PDF sleep signal visualisation report."
    )
    parser.add_argument(
        "-name",
        required=True,
        metavar="PARTICIPANT_FOLDER",
        help='Path to participant folder, e.g. "Data/AP01"',
    )
    args = parser.parse_args()

    folder = args.name.rstrip("/\\")
    if not os.path.isdir(folder):
        sys.exit(f"ERROR: Folder not found: '{folder}'")

    pid = os.path.basename(folder)

    # ── Discover & parse files ────────────────────────────────────────────
    print(f"[vis.py]  Participant : {pid}")
    print(f"[vis.py]  Folder      : {folder}")

    try:
        file_paths = discover_files(folder)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    print("[vis.py]  Parsing files ...")
    nasal,  nasal_hdr  = parse_continuous(file_paths["nasal"])
    thorac, thorac_hdr = parse_continuous(file_paths["thorac"])
    spo2,   spo2_hdr   = parse_continuous(file_paths["spo2"])
    events             = parse_events(file_paths["events"])
    sleep              = parse_sleep(file_paths["sleep"])

    print(f"[vis.py]  Nasal  : {len(nasal):>8,} samples  "
          f"({nasal_hdr.get('Sample Rate','?')} Hz)")
    print(f"[vis.py]  Thorac : {len(thorac):>8,} samples  "
          f"({thorac_hdr.get('Sample Rate','?')} Hz)")
    print(f"[vis.py]  SpO2   : {len(spo2):>8,} samples  "
          f"({spo2_hdr.get('Sample Rate','?')} Hz)")
    print(f"[vis.py]  Events : {len(events)} annotations")
    print(f"[vis.py]  Sleep  : {len(sleep)} epochs")

    # ── Clinical metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(events, sleep, spo2)
    print(f"[vis.py]  AHI    : {metrics['AHI (events/h)']} — {metrics['Severity']}")

    # ── Output path ───────────────────────────────────────────────────────
    out_dir = "Visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pid}_sleep_report.pdf")

    # ── Build PDF ─────────────────────────────────────────────────────────
    print(f"[vis.py]  Generating PDF → {out_path} ...")

    with PdfPages(out_path) as pdf:

        # PDF metadata
        pdf_meta = pdf.infodict()
        pdf_meta["Title"]   = f"Sleep Study Report — {pid}"
        pdf_meta["Author"]  = "vis.py"
        pdf_meta["Subject"] = "Overnight polysomnography signal visualisation"
        pdf_meta["Keywords"] = "sleep, apnea, hypopnea, SpO2, nasal airflow"

        print("[vis.py]    Page 1/7 — Cover ...")
        page_cover(pdf, pid, folder, nasal_hdr, spo2_hdr,
                   events, sleep, metrics)

        print("[vis.py]    Page 2/7 — Full 8-hour overview ...")
        page_overview(pdf, pid, nasal, thorac, spo2, events, sleep)

        print("[vis.py]    Pages 3-5 — 2-hour zoom panels ...")
        pages_zoom(pdf, pid, nasal, thorac, spo2, events, sleep,
                   window_hours=2.0)

        print("[vis.py]    Page 6/7 — SpO2 detail ...")
        page_spo2_detail(pdf, pid, spo2, events)

        print("[vis.py]    Page 7/7 — Event catalogue ...")
        page_event_catalogue(pdf, pid, events)

    print(f"\n[vis.py]  Done.  PDF saved to:  {out_path}")


if __name__ == "__main__":
    main()
