#!/usr/bin/env python3
"""
Rolling-Shutter Wind Sensor (Robust + Hybrid Debug Version)
-----------------------------------------------------------

• Keeps your original detector logic
• Adds:
    - Gray-world colour constancy
    - Strict HSV red gate (AND, not OR → avoids white edges)
    - Anti-glare suppression (bright/low-sat)
    - Morphological cleanup
• Adds:
    - Hybrid debug windows: Debug, Binary ROI, Mask ROI
    - Per-frame quality metrics
    - Per-frame debug CSV logging (RMS(a) vs time, etc.)
• Calibration CSV (RMS + GT) still only logs on new GT
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
import csv
import serial
import re
from collections import deque


# ====================================================
# Utility Functions
# ====================================================

def contrast_stretch(img, low=2, high=98):
    p_low = np.percentile(img, low)
    p_high = np.percentile(img, high)
    scaled = np.clip((img - p_low) * (255.0 / (p_high - p_low + 1e-6)), 0, 255)
    return scaled.astype(np.uint8)


def gamma_correct(img, gamma=1.15):
    inv = 1.0 / gamma
    table = (np.linspace(0, 255, 256) / 255.0) ** inv * 255
    return cv2.LUT(img, table.astype("uint8"))


def gray_world_normalise(frame: np.ndarray) -> np.ndarray:
    """
    Simple gray-world colour constancy.
    Makes channel means similar → more robust to colour cast / lab lighting.
    frame: RGB uint8
    """
    f = frame.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)   # [R_mean, G_mean, B_mean]
    gray_mean = means.mean() + 1e-6
    scale = gray_mean / (means + 1e-6)      # 3 scales

    f *= scale  # broadcast
    f = np.clip(f, 0, 255)
    return f.astype(np.uint8)


def build_robust_red_mask(frame_rgb: np.ndarray):
    """
    Build a *selective* mask for your stripe using:

      1) Gray-world normalisation
      2) Your original "blue" detector (empirically good)
      3) A strict HSV red gate (AND) to avoid non-red (e.g. white edges)

    Returns:
      frame_norm (RGB), hsv, final_mask (uint8 0/255)
    """
    # 1) Colour constancy
    frame_norm = gray_world_normalise(frame_rgb)

    # 2) Colour spaces
    hsv = cv2.cvtColor(frame_norm, cv2.COLOR_RGB2HSV)

    # ------------------------------------------------
    # (A) Original "blue" detector (unchanged logic)
    # ------------------------------------------------
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([130, 255, 255])
    mask_hsv_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    r, g, b = frame_norm[:, :, 0], frame_norm[:, :, 1], frame_norm[:, :, 2]
    strict_blue = (
        (b > 120) &
        (b - g > 40) &
        (b - r > 40) &
        (r < 150) &
        (g < 150)
    ).astype(np.uint8) * 255

    mask_legacy = cv2.bitwise_and(mask_hsv_blue, strict_blue)

    # ------------------------------------------------
    # (B) Strict red gating in HSV
    # ------------------------------------------------
    # Whites / greys have low saturation → kill them via S threshold
    # We allow a bit of flexibility in V so we don't lose darker red.
    lower_red1 = np.array([0,   80, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([170, 80, 40])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red_hsv = cv2.bitwise_or(mask_red1, mask_red2)

    # Extra RGB sanity check: red channel significantly larger than G/B
    rgb_red = (
        (r > g + 15) &
        (r > b + 15)
    ).astype(np.uint8) * 255

    # ------------------------------------------------
    # (C) Final combined mask:
    #     pixel must be:
    #       - in your legacy mask (good structure),
    #       - AND in red hue range,
    #       - AND actually red in RGB.
    # ------------------------------------------------
    mask_strict = cv2.bitwise_and(mask_legacy, mask_red_hsv)
    mask_strict = cv2.bitwise_and(mask_strict, rgb_red)

    # Fallback: if *everything* disappears (extreme weird case),
    # just use legacy (so you don't lose detection completely).
    if np.count_nonzero(mask_strict) < 50:
        combined = mask_legacy.copy()
    else:
        combined = mask_strict

    return frame_norm, hsv, combined


# ====================================================
# Camera Setup
# ====================================================

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam2.configure(config)
picam2.start()

time.sleep(0.5)
for _ in range(6):
    picam2.capture_array()

meta = picam2.capture_metadata()
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": int(meta["ExposureTime"]),
    "AnalogueGain": float(meta["AnalogueGain"])
})

print("Camera locked. Starting sensor...\n")


# ====================================================
# Ground-Truth Serial Setup (STM32)
# ====================================================

GT_PORT = "/dev/ttyACM0"
GT_BAUD = 115200

try:
    gt_ser = serial.Serial(GT_PORT, GT_BAUD, timeout=0.01)
    time.sleep(1)
    print(f"Connected to STM32 GT on {GT_PORT}")
except Exception as e:
    print("\n⚠ Could NOT connect to STM32 GT. GT will be NaN.\n")
    gt_ser = None

last_gt = float("nan")   # saved GT value


# ====================================================
# Parameters
# ====================================================

ROW_START = 150
ROW_END   = 500

MIN_POINTS = 5
RESIDUAL_THRESH = 80.0

RMS_WINDOW = 20
slope_history = deque(maxlen=RMS_WINDOW)


# ====================================================
# CSV LOGGING
# ====================================================

# Calibration CSV: only logs when new GT arrives
log_filename = "calibration_log_sync.csv"
csv_file = open(log_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "rms_slope", "gt_wind"])
print(f"Logging calibration data to {log_filename}")

# Debug CSV: logs EVERY frame for RMS(a) vs time, quality, etc.
debug_filename = "debug_log_all.csv"
debug_file = open(debug_filename, "w", newline="")
debug_writer = csv.writer(debug_file)
debug_writer.writerow([
    "frame_idx",
    "timestamp",
    "rms_slope",
    "num_centroids",
    "rows_with_centroids",
    "roi_rows",
    "quality",
    "resid",
    "gt_wind"
])
print(f"Logging debug data to {debug_filename}\n")


# ====================================================
# MAIN LOOP
# ====================================================

print("Running (press q to quit)...")

frame_idx = 0
last_resid = float("nan")

while True:
    frame_idx += 1

    # --------------------------------------------
    # Capture frame + build robust, selective mask
    # --------------------------------------------
    frame_raw = picam2.capture_array()  # RGB888 from Picamera2
    frame, hsv, mask_all = build_robust_red_mask(frame_raw)

    # --------------------------------------------
    # Anti-glare suppression (bright/low-sat pixels) in ROI
    # --------------------------------------------
    hsv_roi = hsv[ROW_START:ROW_END, :, :]
    v = hsv_roi[:, :, 2]
    s = hsv_roi[:, :, 1]

    glare = ((v > 230) & (s < 40)).astype(np.uint8) * 255
    glare = cv2.dilate(glare, np.ones((5, 5), np.uint8), iterations=1)

    mask_roi = mask_all[ROW_START:ROW_END, :]
    glare_inv = cv2.bitwise_not(glare)
    mask_roi = cv2.bitwise_and(mask_roi, glare_inv)

    # --------------------------------------------
    # Enhance + morphological cleanup
    # --------------------------------------------
    mask_roi_blur = cv2.medianBlur(mask_roi, 5)
    mask_roi_cs   = contrast_stretch(mask_roi_blur)
    mask_roi_gamma = gamma_correct(mask_roi_cs)
    mask_roi_smooth = cv2.GaussianBlur(mask_roi_gamma, (5, 5), 0)

    _, binary = cv2.threshold(mask_roi_smooth, 10, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --------------------------------------------
    # Centroid detection
    # --------------------------------------------
    centroids = []
    h, w = binary.shape

    for i in range(h):
        xs = np.where(binary[i] == 255)[0]
        if len(xs) > 0:
            centroids.append((int(np.mean(xs)), ROW_START + i))

    num_centroids = len(centroids)
    rows_with_centroids = num_centroids
    roi_rows = h
    quality = (rows_with_centroids / roi_rows) if roi_rows > 0 else 0.0

    debug = frame.copy()
    cv2.rectangle(debug, (0, ROW_START), (1280, ROW_END), (255, 0, 0), 2)

    for (x, y) in centroids:
        cv2.circle(debug, (x, y), 4, (0, 0, 255), -1)

    # --------------------------------------------
    # Slope fitting
    # --------------------------------------------
    slope = 0.0
    last_resid = float("nan")

    if num_centroids >= MIN_POINTS:
        ys = np.array([c[1] for c in centroids])
        xs = np.array([c[0] for c in centroids])

        a, b = np.polyfit(ys, xs, 1)
        xs_fit = a * ys + b
        resid = np.mean(np.abs(xs - xs_fit))
        last_resid = float(resid)

        if resid < RESIDUAL_THRESH:
            slope = float(a)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(a * y0 + b), int(a * y1 + b)
            cv2.line(debug, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # --------------------------------------------
    # RMS(a)
    # --------------------------------------------
    slope_history.append(slope)

    if len(slope_history) > 5:
        rms_slope = float(np.sqrt(np.mean(np.square(slope_history))))
    else:
        rms_slope = 0.0

    # --------------------------------------------
    # READ GT (ONLY LOG ON NEW GT)
    # --------------------------------------------
    new_gt_arrived = False

    if gt_ser:
        raw = gt_ser.readline().decode(errors="ignore").strip()

        if raw != "":
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw)

            if match:
                last_gt = float(match.group(0))
                new_gt_arrived = True

    # --------------------------------------------
    # LOG ONLY WHEN GT ARRIVES (Calibration CSV)
    # --------------------------------------------
    if new_gt_arrived:
        ts_cal = time.time()
        csv_writer.writerow([ts_cal, rms_slope, last_gt])
        csv_file.flush()
        print(f"CAL LOGGED → GT={last_gt:.3f}, RMS={rms_slope:.4f}")

    # --------------------------------------------
    # DEBUG LOG (EVERY FRAME)
    # --------------------------------------------
    ts_debug = time.time()
    debug_writer.writerow([
        frame_idx,
        ts_debug,
        rms_slope,
        num_centroids,
        rows_with_centroids,
        roi_rows,
        quality,
        last_resid,
        last_gt
    ])
    # Flush occasionally so you don't lose much if you kill the program
    if frame_idx % 10 == 0:
        debug_file.flush()

    # --------------------------------------------
    # DISPLAY (Hybrid Debug: Debug, Binary ROI, Mask ROI)
    # --------------------------------------------
    quality_pct = quality * 100.0
    resid_text = f"{last_resid:.1f}" if not np.isnan(last_resid) else "n/a"
    slope_text = f"{slope:.5f}"

    cv2.putText(debug, f"RMS(a): {rms_slope:.5f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    cv2.putText(debug, f"Slope: {slope_text}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
    cv2.putText(debug, f"Qual: {rows_with_centroids}/{roi_rows} rows ({quality_pct:.1f}%)",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(debug, f"Resid: {resid_text}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    if np.isnan(last_gt):
        gt_text = "GT: N/A"
    else:
        gt_text = f"GT: {last_gt:.3f} m/s"

    cv2.putText(debug, gt_text,
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)

    # Windows:
    cv2.imshow("Rolling-Shutter Wind Sensor", debug)
    cv2.imshow("Binary ROI", binary)
    cv2.imshow("Mask ROI", mask_roi_smooth)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ====================================================
# Cleanup
# ====================================================
csv_file.close()
debug_file.close()
picam2.stop()
cv2.destroyAllWindows()
if gt_ser:
    gt_ser.close()

print("Finished.")
