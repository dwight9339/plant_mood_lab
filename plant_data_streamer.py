"""
Real‑time plant‑signal visualizer + OSC streamer
------------------------------------------------
• Reads a CSV with columns `Time` and `Value` *or* (future) live Home‑Assistant
  websocket/REST source.
• Displays the current data window in an interactive PyQtGraph GUI.
• Computes basic features each update and shows them in the GUI.
• Streams those features (and optional raw window) to Pure Data via OSC so the
  S.T.E.M. module can generate music.

Usage (CSV test mode)
---------------------
python plant_data_streamer.py \
       --csv pothos.csv \
       --start "2025-04-26T19:30:14.814Z" \
       --window 300             # seconds (5 min) \
       --step 5                 # seconds to slide window each tick \
       --tick 1                 # GUI/OSC update rate in seconds

Hit the spacebar to *pause/resume* window‑scrolling. Use ← / → to step one
window backward/forward while paused.

Dependencies
------------
• pandas, numpy
• pyqtgraph, PyQt5 (or PySide2)
• python‑osc
Install via:
  pip install pandas numpy pyqtgraph PyQt5 python‑osc

Pure Data setup
---------------
netreceive 8000 1 → oscparse → route /feature ...
Route `/window/data` for raw‑sample bursts (index, value pairs)
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pythonosc import udp_client

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets  # <-- QtWidgets gives QMainWindow etc.

# ----------------------------- OSC utility ----------------------------- #

def make_osc_client(host: str, port: int):
    return udp_client.SimpleUDPClient(host, port)


def send_features(client: udp_client.SimpleUDPClient, feat: dict):
    """Send each feature as /feature/<name> <value> (float)."""
    for k, v in feat.items():
        try:
            client.send_message(f"/feature/{k}", float(v))
        except Exception as e:
            print(f"OSC send error {k}: {e}")


def send_window(client: udp_client.SimpleUDPClient, window_vals):
    """Stream index‑value pairs so Pd can fill an array."""
    for i, val in enumerate(window_vals):
        client.send_message("/window/data", [i, float(val)])

# --------------------------- Feature helpers --------------------------- #

def compute_features(ts: np.ndarray):
    if len(ts) == 0:
        return {}
    hist, _ = np.histogram(ts, bins=10, density=True)
    hist = hist + 1e-12  # avoid log(0)
    entropy = -np.sum(hist * np.log2(hist))
    zero_x = int(((ts[:-1] * ts[1:]) < 0).sum())
    return {
        "mean": ts.mean(),
        "std_dev": ts.std(),
        "min": ts.min(),
        "max": ts.max(),
        "range": ts.max() - ts.min(),
        "entropy": entropy,
        "zero_x": zero_x,
    }

# ---------------------------- Main window ----------------------------- #
class PlantWindow(QtWidgets.QMainWindow):  # <-- Switched to QtWidgets
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Plant Data Streamer")
        self.resize(1000, 600)

        # OSC client
        self.osc = make_osc_client(args.osc_host, args.osc_port)

        # Load CSV once for test mode
        if args.csv:
            df = pd.read_csv(args.csv, parse_dates=["last_changed"])
            self.times = (
                pd.to_datetime(df["last_changed"]).values.astype("int64") / 1e9  # seconds
            )
            self.values = df["state"].to_numpy(dtype=float)
        else:
            self.times = np.array([])
            self.values = np.array([])

        # Index cursor into self.values
        self.cursor = 0
        if args.csv and args.start:
            start_ts = pd.to_datetime(args.start).value / 1e9
            self.cursor = int(np.searchsorted(self.times, start_ts))

        # Plot widget
        self.plot = pg.PlotWidget(title="Raw Signal Window")
        self.curve = self.plot.plot(pen=pg.mkPen("#2ecc71", width=2))
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("left", "Value")
        self.plot.setLabel("bottom", "Samples (relative)")
        self.mean_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#e74c3c', width=1, style=QtCore.Qt.DashLine))
        self.plot.addItem(self.mean_line)
        self.zero_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#95a5a6', width=1))
        self.plot.addItem(self.zero_line)

        # Feature text widget
        self.text = pg.TextItem(anchor=(0, 0))
        self.plot.addItem(self.text)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.plot)
        self.setCentralWidget(central)

        self.entropy_plot = pg.PlotWidget(maximumHeight=60)
        self.entropy_curve = self.entropy_plot.plot(pen=pg.mkPen('#f1c40f'))
        layout.addWidget(self.entropy_plot)
        self.ent_history = collections.deque(maxlen=300)

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.timer.start(int(args.tick * 1000))

        # State
        self.paused = False

    # ------------------------- event handlers ------------------------ #
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.paused = not self.paused
        elif event.key() == QtCore.Qt.Key_Right and self.paused:
            self.cursor += int(self.args.step * self.args.sampling_rate)
        elif event.key() == QtCore.Qt.Key_Left and self.paused:
            self.cursor -= int(self.args.step * self.args.sampling_rate)
        self.cursor = np.clip(self.cursor, 0, max(0, len(self.values) - 1))

    # --------------------------- core loop --------------------------- #
    def update_stream(self):
        if not self.args.csv:
            return  # live mode to be implemented later
        if not self.paused:
            self.cursor += int(self.args.step * self.args.sampling_rate)
        win_samples = int(self.args.window * self.args.sampling_rate)
        start = max(0, self.cursor)
        end = min(start + win_samples, len(self.values))
        window_vals = self.values[start:end]
        if window_vals.size == 0:
            return

        # Update plot data
        self.curve.setData(np.arange(window_vals.size), window_vals)

        # Auto-range Y with a bit of head-room so peaks aren’t glued to the border
        lo, hi = window_vals.min(), window_vals.max()
        pad = 0.05 * (hi - lo if hi != lo else 1.0)
        self.plot.setYRange(lo - pad, hi + pad, padding=0)

        # Optional: keep X exactly one window wide
        self.plot.setXRange(0, window_vals.size - 1, padding=0)

        # Compute + display features
        feat = compute_features(window_vals)
        feat_lines = [f"{k}: {v:.3f}" for k, v in feat.items()]
        self.text.setText("\n".join(feat_lines))
        self.mean_line.setValue(feat["mean"])
        self.zero_line.setValue(0)          # handy midline reference
        self.ent_history.append(feat["entropy"])
        self.entropy_curve.setData(self.ent_history)

        # Send OSC
        send_features(self.osc, feat)
        if self.args.send_raw:
            send_window(self.osc, window_vals)

# ----------------------------- CLI glue ----------------------------- #

def cli():
    parser = argparse.ArgumentParser(description="Plant data visualizer + OSC streamer")
    parser.add_argument("--csv", type=Path, help="CSV with Time,Value columns", default=None)
    parser.add_argument("--start", type=str, help="ISO start time in CSV", default=None)
    parser.add_argument("--window", type=int, default=300, help="Window size in seconds (default 300)")
    parser.add_argument("--step", type=int, default=5, help="Seconds to slide window each tick (default 5)")
    parser.add_argument("--tick", type=float, default=1.0, help="GUI/OSC update rate in sec (default 1)")
    parser.add_argument("--sampling_rate", type=float, default=1.0, help="Samples per second in CSV (default 1)")
    parser.add_argument("--osc_host", type=str, default="127.0.0.1", help="OSC target host (Pd)")
    parser.add_argument("--osc_port", type=int, default=8000, help="OSC target port (Pd)")
    parser.add_argument("--send_raw", action="store_true", help="Also stream raw window /window/data")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)  # <-- QtWidgets QApplication
    w = PlantWindow(args)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    cli()