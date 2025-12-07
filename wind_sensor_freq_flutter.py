#!/usr/bin/env python3
"""
Non-Contact Wind Sensor — AC Flicker Proof Version + LOGGING

LOGGED VALUES (CSV):
    timestamp,
    amplitude_px,
    frequency_hz,
    wind_mps,
    wavelength_px   # (derived from frequency)
"""

import time
import csv
import numpy as np
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d


class FlagFlutterSensor:
    """
    Rolling-shutter wind sensor:
    Extracts x(y) waveform of a vertical marker line,
    computes frequency via FFT,
    maps frequency → wind speed using calibration.
    """

    # -------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------
    def __init__(self, camera_index: int = 0, use_picamera: bool = True):
        self.use_picamera = use_picamera
        self.camera_index = camera_index
        self.camera = None
        self.roi = None
        self.roi_set = False

        # Frequency → wind speed slope
        self.calibration_slope = 100.0

        # Minimum lateral movement required to count as wind
        self.amplitude_threshold = 4.5

        # CSV LOG FILE
        self.log_filename = "wind_log.csv"
        self.log_file = open(self.log_filename, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(
            ["timestamp", "amplitude_px", "frequency_hz", "wind_mps", "wavelength_px"]
        )

        print(f"Logging to: {self.log_filename}")

        self._initialize_camera()

    # -------------------------------------------------------------
    # CAMERA SETUP
    # -------------------------------------------------------------
    def _initialize_camera(self) -> None:
        """Initialise Picamera2 if available; otherwise OpenCV."""
        if self.use_picamera:
            try:
                from picamera2 import Picamera2

                self.camera = Picamera2()
                config = self.camera.create_video_configuration(
                    main={"size": (1280, 720), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()

                time.sleep(0.5)
                for _ in range(6):
                    _ = self.camera.capture_array()

                meta = self.camera.capture_metadata()
                self.camera.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": int(meta["ExposureTime"]),
                    "AnalogueGain": float(meta["AnalogueGain"]),
                })

                print("Picamera2 initialised (RGB888).")

            except ImportError:
                print("Picamera2 not found — using OpenCV.")
                self.use_picamera = False
                self._initialize_opencv_camera()
        else:
            self._initialize_opencv_camera()

    def _initialize_opencv_camera(self) -> None:
        """Initialise USB camera."""
        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("OpenCV USB camera initialised.")

    def capture_frame(self) -> np.ndarray:
        """Return a frame from camera."""
        if self.use_picamera:
            return self.camera.capture_array()

        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Capture failed")
        return frame

    # -------------------------------------------------------------
    # ROI HANDLING
    # -------------------------------------------------------------
    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Save region of interest."""
        self.roi = (x, y, w, h)
        self.roi_set = True

    def select_roi_interactive(self, frame: np.ndarray) -> None:
        """User draws a bounding box around marker line."""
        print("Draw ROI around the black marker line, then press ENTER.")
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        if roi[2] > 0:
            self.set_roi(*map(int, roi))
        else:
            self.roi_set = False

    # -------------------------------------------------------------
    # 1. VISION: SUB-PIXEL EDGE TRACKING (GAUSSIAN SMOOTHED)
    # -------------------------------------------------------------
    def extract_edge_profile(self, frame: np.ndarray) -> np.ndarray:
        if not self.roi_set:
            raise ValueError("ROI not set.")

        x0, y0, w, h = self.roi
        roi = frame[y0:y0 + h, x0:x0 + w]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2GRAY if self.use_picamera else cv2.COLOR_BGR2GRAY
        )

        # Step A — global lock using vertical sharpness (Sobel X)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        col_strength = np.sum(np.abs(sobel_x), axis=0)
        lock_x = int(np.argmax(col_strength))

        # Step B — restrict to ±30 px window
        radius = 30
        x_start = max(0, lock_x - radius)
        x_end = min(w, lock_x + radius)
        roi_narrow = gray[:, x_start:x_end]

        # Step C — sub-pixel center-of-mass in inverted intensity
        inverted = np.power(255 - roi_narrow, 2)
        x_positions = np.zeros(h)
        indices = np.arange(roi_narrow.shape[1])

        for r in range(h):
            row = inverted[r]
            total = np.sum(row)
            if total > 0:
                x_positions[r] = x_start + (np.sum(row * indices) / total)
            else:
                x_positions[r] = x_positions[r - 1] if r > 0 else lock_x

        # Step D — Gaussian smoothing to kill 50/60 Hz flicker ripple
        x_positions = gaussian_filter1d(x_positions, sigma=4)

        return x_positions

    # -------------------------------------------------------------
    # 2. PHYSICS: FFT + AMPLITUDE
    # -------------------------------------------------------------
    def calculate_metrics(self, x_positions: np.ndarray) -> tuple[float, float, float]:
        """Return (dom_freq_Hz, amplitude_px, wavelength_px)."""
        if len(x_positions) < 8:
            return 0.0, 0.0, 0.0

        detrended = signal.detrend(x_positions)
        amplitude = float(np.std(detrended))

        # Add Hann window + ZERO PADDING to 4096 points
        windowed = detrended * signal.windows.hann(len(detrended))
        fft_vals = np.abs(fft(windowed, n=4096))
        freqs = fftfreq(4096, d=1.0)

        mask = freqs > 0
        pos_freqs = freqs[mask]
        pos_fft = fft_vals[mask]

        if pos_fft.size == 0:
            return 0.0, amplitude, 0.0

        peak_idx = int(np.argmax(pos_fft))
        dom_freq = float(pos_freqs[peak_idx])

        # Convert frequency (cycles / pixel) → wavelength in pixels
        wavelength = 1.0 / dom_freq if dom_freq > 0 else 0.0

        return dom_freq, amplitude, wavelength

    def estimate_wind_speed(self, freq: float, amplitude: float) -> float:
        """Amplitude gate prevents false positives."""
        if amplitude < self.amplitude_threshold:
            return 0.0
        return self.calibration_slope * freq

    # -------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------
    def run_continuous(self, display: bool = True) -> None:
        print("Wind Sensor Ready")

        try:
            while True:
                frame = self.capture_frame()

                if not self.roi_set:
                    self.select_roi_interactive(frame)
                    continue

                # Extract motion waveform
                x_positions = self.extract_edge_profile(frame)

                # Frequency, amplitude, wavelength
                freq, amp, wavelength = self.calculate_metrics(x_positions)

                # Wind speed estimation
                wind = self.estimate_wind_speed(freq, amp)

                # --- CSV LOGGING ---
                timestamp = time.time()
                self.log_writer.writerow(
                    [timestamp, amp, freq, wind, wavelength]
                )
                self.log_file.flush()

                # Console display
                status = "CALM" if wind == 0 else "WINDY"
                print(
                    f"\r[{time.strftime('%H:%M:%S')}] "
                    f"Amp={amp:.2f}px | "
                    f"Freq={freq:.4f} Hz | "
                    f"Wavelength={wavelength:.2f}px | "
                    f"Wind={wind:.2f} m/s [{status}]     ",
                    end="",
                    flush=True,
                )

                # Visualisation
                if display:
                    vis = frame.copy()
                    x, y, w, h = self.roi

                    cv2.rectangle(vis, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)

                    pts = [(int(x + xp), y + r)
                           for r, xp in enumerate(x_positions)]
                    cv2.polylines(vis, [np.array(pts)],
                                  False, (0, 0, 255), 2)

                    cv2.imshow("Wind Sensor", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi_set = False

        except KeyboardInterrupt:
            print("\nStopped by user.")

        finally:
            self.cleanup()

    # -------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------
    def cleanup(self) -> None:
        """Release camera + close CSV + destroy windows."""
        if self.use_picamera:
            try:
                self.camera.stop()
            except Exception:
                pass
        else:
            self.camera.release()

        self.log_file.close()
        cv2.destroyAllWindows()


def main() -> None:
    sensor = FlagFlutterSensor(use_picamera=True)
    sensor.run_continuous(display=True)


if __name__ == "__main__":
    main()
