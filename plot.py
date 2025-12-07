import matplotlib.pyplot as plt
import numpy as np
import csv

ts = []
rms = []
raw = []
smooth = []

with open("wind_log_fan5.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts.append(float(row["timestamp"]))
        rms.append(float(row["rms_slope"]))
        raw.append(float(row["raw_wind"]))
        smooth.append(float(row["smooth_wind"]))

ts = np.array(ts)
rms = np.array(rms)

plt.figure(figsize=(12,5))
plt.plot(ts - ts[0], rms, label="RMS(a)")
plt.xlabel("Time (s)")
plt.ylabel("RMS Slope")
plt.legend()
plt.grid(True)
plt.show()
