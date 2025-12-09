import matplotlib.pyplot as plt
import numpy as np
import csv

ts = []
rms = []
gt = []

with open("wind_sensor_20251207_213718.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts.append(float(row["timestamp"]))
        rms.append(float(row["rms_slope"]))
        gt.append(float(row["gt_wind"]))  # may be NaN during runs without GT

ts = np.array(ts)
rms = np.array(rms)
gt = np.array(gt)

# --------------------------------------------------------
# PLOT 1: RMS(a) vs Time
# --------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(ts - ts[0], rms, label="RMS(a)")
plt.xlabel("Time (s)")
plt.ylabel("RMS Slope")
plt.title("RMS(a) Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# OPTIONAL PLOT 2: GT vs time (only meaningful when GT present)
# --------------------------------------------------------
valid_gt = ~np.isnan(gt)
if np.any(valid_gt):
    plt.figure(figsize=(12, 5))
    plt.plot(ts[valid_gt] - ts[0], gt[valid_gt], label="Ground Truth (GT)", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Wind Speed (m/s)")
    plt.title("Ground Truth Wind Speed Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
