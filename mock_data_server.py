import time
import numpy as np
from collections import deque
import threading
from flask import Flask, jsonify, request
import tkinter as tk
from tkinter import ttk

# ----------------------------------------
# Signal Simulator
# ----------------------------------------

class SignalSimulator:
    def __init__(self, rate=1.0, max_len=600):
        self.rate = rate            # samples per second
        self.max_len = max_len      # max buffer length (e.g. 600s = 10 min)
        self.buffer = deque(maxlen=max_len)
        self.last_time = time.time()

        # Control params
        self.base = 1.65
        self.amp = 0.1
        self.jitter = 0.1
        self.drift = 0.001
        self.phase = 0.0

    def update(self):
        now = time.time()
        elapsed = now - self.last_time
        step_time = 1.0 / self.rate

        while elapsed >= step_time:
            self.phase += self.drift
            value = (
                self.base +
                self.amp * np.sin(self.phase) +
                np.random.uniform(-self.jitter, self.jitter)
            )
            timestamp = self.last_time + step_time
            self.buffer.append((timestamp, value))
            self.last_time += step_time
            elapsed -= step_time

    def get_window(self, n=300):
        self.update()
        return list(self.buffer)[-n:]

# ----------------------------------------
# GUI Controls
# ----------------------------------------

class SignalGUI:
    def __init__(self, sim: SignalSimulator):
        self.sim = sim
        self.root = tk.Tk()
        self.root.title("Mock Signal Controls")

        self._add_slider("Base", 0.0, 3.3, sim.base, lambda v: setattr(sim, 'base', float(v)))
        self._add_slider("Amplitude", 0.0, 0.5, sim.amp, lambda v: setattr(sim, 'amp', float(v)))
        self._add_slider("Jitter", 0.0, 0.4, sim.jitter, lambda v: setattr(sim, 'jitter', float(v)))
        self._add_slider("Drift", 0.0, 0.01, sim.drift, lambda v: setattr(sim, 'drift', float(v)))

    def _add_slider(self, label, frm, to, initial, command):
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", padx=5, pady=4)
        ttk.Label(frame, text=label).pack(side="left")
        scale = ttk.Scale(frame, from_=frm, to=to, value=initial, orient="horizontal", command=command)
        scale.pack(side="right", fill="x", expand=True)

    def run(self):
        self.root.mainloop()

# ----------------------------------------
# Flask API
# ----------------------------------------

def create_app(sim: SignalSimulator):
    app = Flask(__name__)

    @app.route("/data")
    def serve_data():
        sim.update()
        n = int(request.args.get("window", 300))
        window = sim.get_window(n)

        data = [
            {
                "value": round(val, 6),
                "time": time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(ts))
            }
            for ts, val in window
        ]
        return jsonify(data)

    return app

# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    sim = SignalSimulator(rate=1.0, max_len=1200)  # 20 minutes at 1Hz
    app = create_app(sim)

    # Start Flask server in background
    threading.Thread(target=lambda: app.run(port=5000), daemon=True).start()

    # Launch GUI
    gui = SignalGUI(sim)
    gui.run()
