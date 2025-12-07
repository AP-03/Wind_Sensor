#!/usr/bin/env python3
"""
Rolling-Shutter Wind Sensor (Final UART Version for ttyAMA10)
- Robust red-stripe detection
- Computes RMS(a)
- Applies calibration: v_est = k * RMS(a) + b
- Streams: timestamp, RMS(a), v_est  --> UART TX
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import time
import csv
import serial
import re
from collections import deque

# ======================================================
# UART OUT (Pi TX -> STM32 RX)
# ======================================================
UART_OUT = serial.Serial("/dev/ttyAMA0", 115200, timeout=0)
print("[OK] UART OUT open on /dev/ttyAMA0")

# ======================================================
# OPTIONAL Ground Truth Input (STM32)
# ======================================================
GT_PORT = "/dev/ttyACM0"
GT_BAUD = 115200
try:
    gt_ser = serial.Serial(GT_PORT, GT_BAUD, timeout=0.01)
    print("[OK] Connected to GT (STM32)")
except:
    print("[WARN] No GT connected.")
    gt_ser = None

last_gt = float("nan")

# ======================================================
# CALIBRATION PARAMETERS (FILL AFTER TUNNEL TEST)
# ======================================================
GAIN = 1.0
BIAS = 0.0

# ======================================================
# Utility Functions
# ======================================================
def contrast_stretch(img):
    p_low = np.percentile(img, 2)
    p_high = np.percentile(img, 98)
    scaled = np.clip((img - p_low) * 255.0 / (p_high - p_low + 1e-6), 0, 255)
    return scaled.astype(np.uint8)

def gamma_correct(img, gamma=1.15):
    inv = 1.0 / gamma
    table = (np.linspace(0,255,256)/255.0)**inv * 255
    return cv2.LUT(img, table.astype("uint8"))

def gray_world(img):
    f = img.astype(np.float32)
    means = f.reshape(-1,3).mean(axis=0)
    gm = means.mean() + 1e-6
    scale = gm/(means+1e-6)
    f *= scale
    return np.clip(f,0,255).astype(np.uint8)

def build_mask(rgb):
    norm = gray_world(rgb)
    hsv = cv2.cvtColor(norm, cv2.COLOR_RGB2HSV)

    lower_blue = np.array([90,60,40])
    upper_blue = np.array([130,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    r,g,b = norm[:,:,0], norm[:,:,1], norm[:,:,2]
    strict = ((b>120)&(b-g>40)&(b-r>40)&(r<150)&(g<150)).astype(np.uint8)*255
    legacy = cv2.bitwise_and(mask_blue, strict)

    mask_r1 = cv2.inRange(hsv, np.array([0,80,40]), np.array([15,255,255]))
    mask_r2 = cv2.inRange(hsv, np.array([170,80,40]), np.array([180,255,255]))
    hsv_red = cv2.bitwise_or(mask_r1, mask_r2)

    rgb_red = ((r>g+15)&(r>b+15)).astype(np.uint8)*255

    combined = cv2.bitwise_and(legacy, hsv_red)
    combined = cv2.bitwise_and(combined, rgb_red)

    if np.count_nonzero(combined) < 50:
        combined = legacy

    return norm, hsv, combined

# ======================================================
# Camera
# ======================================================
picam = Picamera2()
config = picam.create_video_configuration(
    main={"format":"RGB888","size":(1280,720)}
)
picam.configure(config)
picam.start()

time.sleep(0.5)
for _ in range(5):
    picam.capture_array()

meta = picam.capture_metadata()
picam.set_controls({
    "AeEnable":False,
    "AwbEnable":False,
    "ExposureTime":int(meta["ExposureTime"]),
    "AnalogueGain":float(meta["AnalogueGain"])
})

print("[OK] Camera ready.")

# ======================================================
# Calibration Log
# ======================================================
csv_file = open("calibration_log_sync.csv","w",newline="")
cw = csv.writer(csv_file)
cw.writerow(["timestamp", "rms_slope", "gt"])

# ======================================================
# Main Loop
# ======================================================
ROW_START = 150
ROW_END = 500
MIN_POINTS = 5
RESID_THRESH = 80.0

hist = deque(maxlen=60)

print("Running...")

while True:

    # 1. Capture
    raw = picam.capture_array()
    frame, hsv, mask = build_mask(raw)

    # ROI
    hsv_roi = hsv[ROW_START:ROW_END]
    v = hsv_roi[:,:,2]
    s = hsv_roi[:,:,1]
    glare = ((v>230)&(s<40)).astype(np.uint8)*255
    glare = cv2.dilate(glare, np.ones((5,5),np.uint8),1)

    mask_roi = mask[ROW_START:ROW_END]
    mask_roi = cv2.bitwise_and(mask_roi, cv2.bitwise_not(glare))

    mask_roi = cv2.medianBlur(mask_roi,5)
    mask_roi = contrast_stretch(mask_roi)
    mask_roi = gamma_correct(mask_roi)
    mask_roi = cv2.GaussianBlur(mask_roi,(5,5),0)
    _, binary = cv2.threshold(mask_roi,10,255,cv2.THRESH_BINARY)

    # 2. Centroids
    centroids = []
    for i in range(binary.shape[0]):
        xs = np.where(binary[i]==255)[0]
        if len(xs)>0:
            centroids.append((int(xs.mean()), ROW_START+i))

    # 3. Slope
    slope = 0.0
    if len(centroids)>=MIN_POINTS:
        ys = np.array([c[1] for c in centroids])
        xs = np.array([c[0] for c in centroids])

        a,b = np.polyfit(ys,xs,1)
        resid = np.mean(np.abs(xs-(a*ys + b)))
        if resid < RESID_THRESH:
            slope = float(a)

    # 4. RMS
    hist.append(slope)
    if len(hist)>5:
        rms = float(np.sqrt(np.mean(np.square(hist))))
    else:
        rms = 0.0

    # 5. GT read
    new_gt = False
    if gt_ser:
        raw_gt = gt_ser.readline().decode("ignore").strip()
        if raw_gt:
            m = re.search(r"[-+]?\d*\.\d+|\d+", raw_gt)
            if m:
                last_gt = float(m.group(0))
                new_gt = True

    if new_gt:
        ts = time.time()
        cw.writerow([ts, rms, last_gt])
        csv_file.flush()

    # 6. Calibration
    v_est = GAIN * rms + BIAS

    # 7. UART OUT
    msg = f"{time.time():.3f},{rms:.5f},{v_est:.3f}\n"
    UART_OUT.write(msg.encode())

    if cv2.waitKey(1)&0xFF == ord('q'):
        break

csv_file.close()
picam.stop()
UART_OUT.close()
if gt_ser:
    gt_ser.close()

print("Finished.")
