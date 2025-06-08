"""
Real-time plant-signal visualiser + OSC streamer  (2025-05-28)
────────────────────────────────────────────────────────────────
Change log (2025-05-28):
  • Fix velocity-pattern trace: switch to stepMode="center" so X and Y arrays
    can be the same length and the deprecation warning disappears.
  • Reduce auxiliary timeline height by giving the main plot a higher stretch
    factor (3:1 ratio).
  • Kept all strings in double quotes as requested.
"""

import argparse
from pathlib import Path
import sys
from typing import Sequence
import requests

import numpy as np
import pandas as pd
from pythonosc import udp_client
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from scipy.signal import find_peaks, hilbert  # pip install scipy

# ------------------------------------------------------------------ OSC helpers

def make_osc_client(host: str, port: int):
    return udp_client.SimpleUDPClient(host, port)


def _is_sequence(x):
    return isinstance(x, (list, tuple, np.ndarray))


def send_features(client: udp_client.SimpleUDPClient, feat: dict):
    """Send each feature via OSC – arrays go as multi-arg messages."""
    for k, v in feat.items():
        try:
            if _is_sequence(v):
                client.send_message(f"/feature/{k}", [float(x) for x in v])
            else:
                client.send_message(f"/feature/{k}", float(v))
        except Exception as e:
            print(f"OSC send error on {k}: {e}")


def send_window(client: udp_client.SimpleUDPClient, window_vals: np.ndarray):
    for i, val in enumerate(window_vals):
        client.send_message("/window/data", [i, float(val)])

# -------------------------------------------------------------- helpers

def _rolling_mean(ts: np.ndarray, window: int) -> np.ndarray:
    if ts.size == 0 or window < 1:
        return np.zeros_like(ts)
    kernel = np.ones(window) / window
    return np.convolve(ts, kernel, mode="same")


def _velocity_pattern(vel: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Return an n-length 0/1 array showing which chunks contain salient velocity spikes."""
    if vel.size == 0:
        return np.zeros(n_bins)

    abs_vel = np.abs(vel)
    thr = np.percentile(abs_vel, 75)
    if thr == 0:
        return np.zeros(n_bins)

    flags = []
    edges = np.linspace(0, vel.size, n_bins + 1, dtype=int)
    for i in range(n_bins):
        chunk = abs_vel[edges[i]: edges[i + 1]]
        flags.append(float((chunk > thr).mean() > 0.25))
    return np.array(flags)

class MetaScaler:
    def __init__(self, buffer_size=300):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, val):
        self.buffer.append(val)
        if len(self.buffer) < 2:
            return 0.0
        min_val = min(self.buffer)
        max_val = max(self.buffer)
        return (val - min_val) / (max_val - min_val + 1e-6)

scalers = {
    "mean": MetaScaler(),
    "std": MetaScaler(),
    "roll64": MetaScaler(),
    "mean_vel": MetaScaler(),
    "rms_energy": MetaScaler(),
    "entropy": MetaScaler(),
    "hilb_env": MetaScaler()
}

def compute_features(ts: np.ndarray, fs: float = 1.0) -> dict:
    """Return a handful of scalar & vector features for downstream use."""
    if ts.size == 0:
        return {}

    ts_center = ts - ts.mean()
    vel = np.diff(ts, prepend=ts[0])

    raw_mean = float(ts.mean())
    mean = scalers["mean"].update(raw_mean)
    raw_std = float(ts.std())
    std = scalers["std"].update(raw_std)
    raw_roll64 = float(_rolling_mean(ts, 64)[-1])
    roll64 = scalers["roll64"].update(raw_roll64)
    raw_mean_vel = float(vel.mean())
    mean_vel = scalers["mean_vel"].update(raw_mean_vel)
    raw_rms_energy = float(np.sqrt((ts ** 2).mean()))
    rms_energy = scalers["rms_energy"].update(raw_rms_energy)

    hist, _ = np.histogram(ts, bins=10, density=False)
    hist = hist / hist.sum()
    raw_entropy = float(-np.sum((hist + 1e-12) * np.log2(hist + 1e-12)))
    entropy = scalers["entropy"].update(raw_entropy)

    peaks, _ = find_peaks(ts)
    peak_intervals = np.diff(peaks) if peaks.size > 1 else np.array([])

    ts_ds = ts_center[::5]
    acf = np.correlate(ts_ds, ts_ds, mode="full")[len(ts_ds):]
    acf = acf / acf[0] if acf[0] else acf
    acf_lag = int(np.argmax(acf[2:]) + 2)

    fft_mag = np.abs(np.fft.rfft(ts_center, n=4096))
    freqs = np.fft.rfftfreq(4096, d=1.0 / fs)

    if fft_mag[1:].sum() == 0:
        spectral_centroid = 0.0
        pitch_bin = 0.0
    else:
        raw_spectral_centroid = float((freqs[1:] * fft_mag[1:]).sum() / fft_mag[1:].sum())
        spectral_centroid = min(max(raw_spectral_centroid / (fs / 2), 0.0), 1.0)
        dominant_freq = freqs[1:][np.argmax(fft_mag[1:])]
        if dominant_freq > 0:
            midi_note = 69 + 12 * np.log2(dominant_freq / 440.0)
            pitch_bin = int(round(midi_note)) % 12
        else:
            pitch_bin = 0

    raw_hilb_env = float(np.abs(hilbert(ts))[-1])
    hilb_env = scalers["hilb_env"].update(raw_hilb_env)
    vel_pattern = _velocity_pattern(vel, 16)

    return {
        "mean": mean,
        "std_dev": std,
        "rollmean_64": roll64,
        "mean_velocity": mean_vel,
        "rms_energy": rms_energy,
        "entropy": entropy,
        "peak_intervals": peak_intervals,
        "acf_lag": acf_lag,
        "spectral_centroid": spectral_centroid,
        "pitch_bin": pitch_bin,
        "hilbert_envelope": hilb_env,
        "velocity_pattern": vel_pattern,
        "window_size": ts.size,
    }

class InfiniteLineLegendItem(ItemSample):
    def __init__(self, line):
        dummy_item = pg.PlotDataItem(pen=line.pen)
        super().__init__(dummy_item)


# ------------------------------------------------------------- main window

class PlantWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Plant Data Streamer")
        self.resize(1000, 700)

        # OSC
        self.osc = make_osc_client(args.osc_host, args.osc_port)

        # Data source
        if args.csv:
            df = pd.read_csv(args.csv, parse_dates=[args.time_col])
            self.times = pd.to_datetime(df[args.time_col]).values.astype("int64") / 1e9
            self.values = df[args.value_col].to_numpy(dtype=float)
        elif args.api:
            try:
                url = self.args.api + f"?window={int(self.args.window)}"
                print(url)
                r = requests.get(url, timeout=2)
                r.raise_for_status()
                records = r.json()
                self.times = np.array([
                    pd.to_datetime(row[args.time_col]).value / 1e9
                    for row in records
                ])
                self.values = np.array([float(row[args.value_col]) for row in records])
            except Exception as e:
                print(f"Failed to fetch from API: {e}")
        else:
            self.times = np.array([])
            self.values = np.array([])

        # Cursor init
        self.cursor = 0
        if args.csv and args.start:
            start_ts = pd.to_datetime(args.start).value / 1e9
            self.cursor = int(np.searchsorted(self.times, start_ts))

        # ——— GUI ———
        self.plot = pg.PlotWidget(title="Raw Signal + RollMean64")
        self.legend = self.plot.getPlotItem().addLegend(offset=(10, 10))
        self.curve = self.plot.plot(pen=pg.mkPen("#2ecc71", width=2), name="Raw Signal")
        self.roll_curve = self.plot.plot(pen=pg.mkPen("#3498db", width=2), name="RollMean64")
        self.mean_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#e67e22", width=1, style=QtCore.Qt.DashLine), name="Mean")
        self.plot.addItem(self.mean_line)
        self.legend.addItem(InfiniteLineLegendItem(self.mean_line), "Mean")

        self.plot.showGrid(False, False)  # remove grid
        self.plot.setLabel("left", "Value")
        self.plot.setLabel("bottom", "Samples")

        self.text = pg.TextItem(anchor=(0, 0))
        self.plot.addItem(self.text)

        # Auxiliary plot (timeline of derived features)
        self.aux_plot = pg.PlotWidget(title="Peaks & Velocity Pattern")
        self.aux_plot.addLegend(offset=(10, 10))
        self.aux_plot.setXLink(self.plot)
        self.aux_plot.setLabel("left", "Feature map")
        self.aux_plot.setLabel("bottom", "Samples")
        self.aux_plot.setYRange(0.0, 1.0, padding=0)
        self.aux_plot.showGrid(False, False)  # remove grid

        # Items for auxiliary plot
        self.vel_curve = pg.PlotDataItem(pen=pg.mkPen("#9b59b6", width=2), stepMode="center", name="Velocity Pattern")
        self.peaks_scatter = pg.ScatterPlotItem(brush=pg.mkBrush("#e74c3c"), size=6, name="Peaks")
        self.aux_plot.addItem(self.vel_curve)
        self.aux_plot.addItem(self.peaks_scatter)


        # Layout (raw plot 3× height of aux plot)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.plot, stretch=3)
        layout.addWidget(self.aux_plot, stretch=1)
        self.setCentralWidget(central)

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.timer.start(int(args.tick * 1000))
        self.paused = False

    # ——— keyboard ———
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.paused = not self.paused
        elif event.key() == QtCore.Qt.Key_Right and self.paused:
            self.cursor += int(self.args.step * self.args.sampling_rate)
        elif event.key() == QtCore.Qt.Key_Left and self.paused:
            self.cursor -= int(self.args.step * self.args.sampling_rate)
        self.cursor = np.clip(self.cursor, 0, max(0, len(self.values) - 1))

    # ——— main loop ———
    def update_stream(self):
        # --------------------------------------
        # Option 1: API mode – fetch fresh data
        if self.args.api:
            try:
                url = self.args.api + f"?window={int(self.args.window)}"
                r = requests.get(url, timeout=2)
                r.raise_for_status()
                records = r.json()
                times = np.array([
                    pd.to_datetime(row[self.args.time_col]).value / 1e9
                    for row in records
                ])
                values = np.array([float(row[self.args.value_col]) for row in records])
            except Exception as e:
                print(f"[API fetch error] {e}")
                return

            self.times = times
            self.values = values
            self.cursor = max(0, len(values) - int(self.args.window * self.args.sampling_rate))

        # --------------------------------------
        # Option 2: CSV mode – walk forward through file
        elif self.args.csv:
            if not self.paused:
                self.cursor += int(self.args.step * self.args.sampling_rate)

        # --------------------------------------
        # Get current window
        win_samps = int(self.args.window * self.args.sampling_rate)
        start = max(0, self.cursor)
        end = min(start + win_samps, len(self.values))
        window_vals = self.values[start:end]
        if window_vals.size == 0:
            return

        # ------------ raw‑signal plot ------------
        x = np.arange(window_vals.size)
        self.curve.setData(x, window_vals)
        self.roll_curve.setData(x, _rolling_mean(window_vals, 64))
        self.plot.setXRange(0, window_vals.size - 1, padding=0)

        lo, hi = window_vals.min(), window_vals.max()
        pad = 0.05 * (hi - lo if hi != lo else 1)
        self.plot.setYRange(lo - pad, hi + pad, padding=0)

        # ------------ features + OSC ------------
        feat = compute_features(window_vals, fs=self.args.sampling_rate)
        self.mean_line.setPos(feat.get("mean", 0.0))

        lines = [f"{k}: {v:.3f}" for k, v in feat.items() if not _is_sequence(v)]
        self.text.setText("\n".join(lines[:8]))

        send_features(self.osc, feat)
        if self.args.send_raw:
            send_window(self.osc, window_vals)

        # ------------ auxiliary plot ------------
        peaks, _ = find_peaks(window_vals)
        if peaks.size:
            self.peaks_scatter.setData(peaks, np.full_like(peaks, 0.9))
        else:
            self.peaks_scatter.setData([], [])

        vel_pattern = feat.get("velocity_pattern", np.zeros(16))
        n_bins = vel_pattern.size
        if n_bins:
            bin_edges = np.linspace(0, window_vals.size, n_bins + 1)
            vel_y = 0.5 + 0.4 * vel_pattern
            self.vel_curve.setData(bin_edges, vel_y)


# ----------------------------------------------------------- CLI

def cli():
    p = argparse.ArgumentParser(description="Plant data visualiser + OSC streamer")
    p.add_argument("--csv", type=Path, help="CSV with last_changed,state", default=None)
    p.add_argument("--start", type=str, help="ISO8601 start timestamp", default=None)
    p.add_argument("--window", type=float, default=300.0, help="Window length (s)")
    p.add_argument("--step", type=float, default=5.0, help="Window hop (s) per tick")
    p.add_argument("--tick", type=float, default=1.0, help="GUI/OSC update rate (s)")
    p.add_argument("--api", type=str, help="URL to fetch mock or live signal data")
    p.add_argument("--sampling_rate", type=float, default=1.0, help="Samples per second")
    p.add_argument("--osc_host", type=str, default="127.0.0.1", help="OSC host")
    p.add_argument("--osc_port", type=int, default=8000, help="OSC port")
    p.add_argument("--send_raw", action="store_true", help="Also stream /window/data")
    p.add_argument("--value_col", default="value", help="Key for signal value")
    p.add_argument("--time_col", default="time", help="Key for timestamp")
    args = p.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = PlantWindow(args)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    cli()
