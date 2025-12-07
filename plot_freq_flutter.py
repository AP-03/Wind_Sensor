import matplotlib.pyplot as plt
import numpy as np
import csv

# Lists for CSV columns
ts = []
freqs = []
waves = []
amps = []
winds = []

# -------- LOAD CSV --------
with open("wind_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts.append(float(row["timestamp"]))
        freqs.append(float(row["frequency_hz"]))
        waves.append(float(row["wavelength_px"]))
        amps.append(float(row["amplitude_px"]))
        winds.append(float(row["wind_mps"]))

# Convert to numpy arrays
ts = np.array(ts)
freqs = np.array(freqs)
waves = np.array(waves)
amps = np.array(amps)
winds = np.array(winds)

# Normalize time to start at zero
t = ts - ts[0]

# ----------- PLOT 1: Wind Speed -----------
plt.figure(figsize=(14, 5))
plt.plot(t, winds, label="Wind Speed (m/s)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Wind Speed (m/s)")
plt.title("Wind Speed vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------- PLOT 2: Frequency -----------
plt.figure(figsize=(14, 5))
plt.plot(t, freqs, label="Frequency (Hz)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------- PLOT 3: Amplitude -----------
plt.figure(figsize=(14, 5))
plt.plot(t, amps, label="Amplitude (px)", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (px)")
plt.title("Amplitude vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------- OPTIONAL: Wavelength -----------
plt.figure(figsize=(14, 5))
plt.plot(t, waves, label="Wavelength (px)", color="purple")
plt.xlabel("Time (s)")
plt.ylabel("Wavelength (px)")
plt.title("Wavelength vs Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
