#!/usr/bin/env python3
"""
Robust Flag Wind Sensor - Multi-Point Tracking Version
======================================================

Approach:
1. Track high-contrast markers on flag tip using optical flow
2. Compute multiple metrics:
   - RMS displacement (correlates with wind speed)
   - Flutter frequency via FFT
   - Extension angle (flag deflection)
3. Fuse metrics for wind speed estimate

Flag Design: Place 3-5 black dots/squares near flag tip on white background
"""

import cv2
import numpy as np
import time
import csv
import serial
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List


def kalman_filter_1d(measurements, process_variance=1e-5, measurement_variance=0.01):
    """Simple 1D Kalman filter for smoothing wind speed estimates"""
    n = len(measurements)
    filtered = np.zeros(n)
    
    # Initial state
    x = measurements[0] if not np.isnan(measurements[0]) else 0.0  # state estimate
    P = 1.0  # estimation error covariance
    
    for i in range(n):
        if np.isnan(measurements[i]):
            filtered[i] = x
            continue
            
        # Prediction
        x_pred = x
        P_pred = P + process_variance
        
        # Update
        K = P_pred / (P_pred + measurement_variance)  # Kalman gain
        x = x_pred + K * (measurements[i] - x_pred)
        P = (1 - K) * P_pred
        
        filtered[i] = x
    
    return filtered


def fit_cubic_model(rms_value, coeffs):
    """
    Apply cubic polynomial model: wind_speed = a*rms³ + b*rms² + c*rms + d
    
    Args:
        rms_value: RMS displacement value
        coeffs: Tuple of (a, b, c, d) coefficients
    
    Returns:
        Estimated wind speed in m/s
    """
    a, b, c, d = coeffs
    return a * rms_value**3 + b * rms_value**2 + c * rms_value + d


def fit_logarithmic_model(rms_value, coeffs):
    """
    Apply logarithmic model: wind_speed = a*ln(rms) + b
    
    Args:
        rms_value: RMS displacement value (must be > 0)
        coeffs: Tuple of (a, b) coefficients
    
    Returns:
        Estimated wind speed in m/s
    """
    a, b = coeffs
    if rms_value <= 0:
        return 0.0
    return a * np.log(rms_value) + b


@dataclass
class WindMetrics:
    """Container for computed wind metrics"""
    rms_displacement: float = 0.0
    flutter_freq_hz: float = 0.0
    extension_angle_deg: float = 0.0
    estimated_wind_mps: float = 0.0
    num_tracked_points: int = 0
    quality: float = 0.0


class RobustWindSensor:
    """
    Multi-point optical flow wind sensor.
    
    Works by tracking high-contrast markers on the flag and analyzing
    their motion patterns to estimate wind speed.
    """
    
    def __init__(
        self,
        use_picamera: bool = True,
        frame_size: Tuple[int, int] = (1280, 720),
        target_fps: float = 30.0
    ):
        self.use_picamera = use_picamera
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.camera = None
        
        # ROI for flag tracking
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.pole_x: Optional[int] = None  # X position of flag pole (for angle calc)
        
        # Optical flow parameters (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=20,
            qualityLevel=0.1,
            minDistance=15,
            blockSize=7
        )
        
        # Tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.point_history: List[deque] = []  # History for each tracked point
        self.history_length = 90  # ~3 seconds at 30fps
        
        # Displacement history for RMS calculation
        self.displacement_history = deque(maxlen=30)
        
        # ============================================================
        # METHOD SELECTION - Change this to switch algorithms
        # ============================================================
        # Options:
        #   'cubic_kalman'   - Cubic polynomial + Kalman filter (R²=0.8814)
        #   'log_kalman'     - Logarithmic model + Kalman filter (R²=0.8306)
        #   'robust_filter'  - Legacy linear model + robust filtering
        # ============================================================
        self.method = 'robust_filter'  # <-- CHANGE THIS TO SWITCH METHODS
        # ============================================================
        
        # Cubic model coefficients: wind_speed = a*rms³ + b*rms² + c*rms + d
        # From calibration with R² = 0.8052 (raw), 0.8814 (Kalman filtered)
        self.cubic_coeffs = (0.0186, -0.5154, 3.9319, -0.4072)
        
        # Logarithmic model coefficients: wind_speed = a*ln(rms) + b
        # From calibration with R² = 0.7202 (raw), 0.8306 (Kalman filtered)
        self.log_coeffs = (2.3969, 3.9180)
        
        # Kalman filter parameters (for cubic_kalman and log_kalman methods)
        self.kalman_process_var = 6e-5      # Lower = smoother but slower response
        self.kalman_measurement_var = 0.05  # Lower = trust measurements more (less smooth)
        self.kalman_state = 0.0
        self.kalman_covariance = 1.0
        
        # Legacy linear calibration (for robust_filter method)
        self.cal_rms_gain = 1.2      # m/s per pixel RMS
        self.cal_rms_offset = 0.0
        self.cal_freq_gain = 2.0     # m/s per Hz
        self.cal_angle_gain = 0.1    # m/s per degree
        self.weight_rms = 1.0
        self.weight_freq = 0.0
        self.weight_angle = 0.0
        
        # Robust filter state (for robust_filter method)
        self.filtered_wind = 0.0           # Current filtered output
        self.running_avg = 0.0             # Slow-moving average for outlier detection
        self.filter_alpha = 0.2            # EMA smoothing: Higher = faster response (0.1-0.3)
        self.filter_max_rate = 0.3         # Max m/s change per frame: Higher = faster jumps allowed
        self.filter_outlier_thresh = 1.5   # Outlier rejection: Higher = accept larger jumps (m/s)
        
        # Ground truth serial (input)
        self.gt_serial: Optional[serial.Serial] = None
        self.last_gt = float('nan')
        
        # UART output (to MacBook)
        self.uart_out: Optional[serial.Serial] = None
        
        # CSV logging
        self.log_file = None
        self.log_writer = None
        self.cal_file = None
        self.cal_writer = None
        
        self._init_camera()
        self._init_logging()
    
    def _init_camera(self):
        """Initialize camera with locked exposure"""
        if self.use_picamera:
            try:
                from picamera2 import Picamera2
                
                self.camera = Picamera2()
                config = self.camera.create_video_configuration(
                    main={"size": self.frame_size, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                
                # Warm up and lock exposure
                time.sleep(0.5)
                for _ in range(10):
                    self.camera.capture_array()
                
                meta = self.camera.capture_metadata()
                self.camera.set_controls({
                    "AeEnable": False,
                    "AwbEnable": False,
                    "ExposureTime": int(meta["ExposureTime"]),
                    "AnalogueGain": float(meta["AnalogueGain"])
                })
                print("[OK] PiCamera2 initialized with locked exposure")
                
            except ImportError:
                print("[WARN] PiCamera2 not available, using OpenCV")
                self.use_picamera = False
                self._init_opencv_camera()
        else:
            self._init_opencv_camera()
    
    def _init_opencv_camera(self):
        """Fallback to OpenCV camera"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        print("[OK] OpenCV camera initialized")
    
    def _init_logging(self):
        """Setup CSV logging"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Main sensor log
        log_filename = f"wind_sensor_{timestamp}.csv"
        self.log_file = open(log_filename, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "timestamp", "rms_displacement", "flutter_freq_hz", 
            "extension_angle_deg", "wind_estimate_mps", 
            "num_points", "quality", "gt_wind_mps"
        ])
        print(f"[OK] Logging to {log_filename}")
        
        # Calibration log (only when GT available)
        cal_filename = f"calibration_{timestamp}.csv"
        self.cal_file = open(cal_filename, "w", newline="")
        self.cal_writer = csv.writer(self.cal_file)
        self.cal_writer.writerow([
            "timestamp", "rms_displacement", "flutter_freq_hz",
            "extension_angle_deg", "gt_wind_mps"
        ])
        print(f"[OK] Calibration log: {cal_filename}")
    
    def connect_ground_truth(self, port: str = "/dev/ttyACM0", baudrate: int = 115200):
        """Connect to ground truth wind sensor"""
        try:
            self.gt_serial = serial.Serial(port, baudrate, timeout=0.01)
            time.sleep(1)
            print(f"[OK] Ground truth connected on {port}")
        except Exception as e:
            print(f"[WARN] Could not connect to ground truth: {e}")
            self.gt_serial = None
    
    def connect_uart_output(self, port: str = "/dev/ttyAMA0", baudrate: int = 115200):
        """Setup UART output to MacBook"""
        try:
            self.uart_out = serial.Serial(port, baudrate, timeout=0, write_timeout=2)
            time.sleep(0.5)
            print(f"[OK] UART output ready on {port}")
        except Exception as e:
            print(f"[WARN] Could not open UART output: {e}")
            self.uart_out = None
    
    def send_uart(self, metrics: WindMetrics):
        """Send wind estimate over UART to MacBook"""
        if self.uart_out is None:
            return
        
        # Only send wind speed estimation (matches CameraTest.py format)
        msg = f"{metrics.estimated_wind_mps:.3f}\r\n"
        
        try:
            self.uart_out.write(msg.encode())
            self.uart_out.flush()
        except serial.SerialTimeoutException:
            pass  # Don't block on slow receiver
    
    def capture_frame(self) -> np.ndarray:
        """Capture a frame from the camera"""
        if self.use_picamera:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Frame capture failed")
            return frame
    
    def select_roi(self, frame: np.ndarray):
        """Interactive ROI selection"""
        print("\n[SETUP] Draw a rectangle around the FLAG TIP area")
        print("        Include the markers/dots you want to track")
        print("        Press ENTER when done, or 'c' to cancel\n")
        
        roi = cv2.selectROI("Select Flag Tip Region", frame, fromCenter=False)
        cv2.destroyWindow("Select Flag Tip Region")
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi = tuple(map(int, roi))
            print(f"[OK] ROI set: {self.roi}")
            
            # Ask for pole position
            print("\n[SETUP] Now click on the FLAG POLE position (for angle calculation)")
            print("        Press any key after clicking\n")
            
            pole_pos = []
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    pole_pos.append(x)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Click Pole Position", frame)
            
            cv2.imshow("Click Pole Position", frame)
            cv2.setMouseCallback("Click Pole Position", mouse_callback)
            cv2.waitKey(0)
            cv2.destroyWindow("Click Pole Position")
            
            if pole_pos:
                self.pole_x = pole_pos[0]
                print(f"[OK] Pole X position: {self.pole_x}")
            else:
                self.pole_x = self.roi[0]  # Default to ROI left edge
        else:
            print("[WARN] ROI selection cancelled")
    
    def detect_markers(self, gray_roi: np.ndarray) -> np.ndarray:
        """
        Detect trackable features in ROI.
        For best results, use black dots/squares on white flag.
        """
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_roi)
        
        # Detect corners/features
        corners = cv2.goodFeaturesToTrack(enhanced, **self.feature_params)
        
        if corners is not None:
            return corners.reshape(-1, 2)
        return np.array([])
    
    def track_points(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Track points between frames using optical flow.
        Returns: (current_points, displacements, num_good)
        """
        if self.roi is None:
            return np.array([]), np.array([]), 0
        
        x, y, w, h = self.roi
        gray_roi = gray[y:y+h, x:x+w]
        
        # Initialize tracking points if needed
        if self.prev_points is None or len(self.prev_points) < 3:
            markers = self.detect_markers(gray_roi)
            if len(markers) > 0:
                # Convert to full image coordinates
                self.prev_points = markers + np.array([x, y])
                self.prev_points = self.prev_points.reshape(-1, 1, 2).astype(np.float32)
                
                # Initialize history for each point
                self.point_history = [deque(maxlen=self.history_length) 
                                     for _ in range(len(self.prev_points))]
            else:
                return np.array([]), np.array([]), 0
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.array([]), np.array([]), 0
        
        # Calculate optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if curr_points is None:
            self.prev_points = None
            return np.array([]), np.array([]), 0
        
        # Filter good points
        status = status.flatten()
        good_mask = status == 1
        
        # Additional validation: points should stay in ROI
        for i, (pt, good) in enumerate(zip(curr_points.reshape(-1, 2), good_mask)):
            if good:
                if not (x <= pt[0] <= x + w and y <= pt[1] <= y + h):
                    good_mask[i] = False
        
        good_curr = curr_points[good_mask]
        good_prev = self.prev_points[good_mask]
        
        if len(good_curr) == 0:
            self.prev_points = None
            return np.array([]), np.array([]), 0
        
        # Calculate displacements
        displacements = good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2)
        
        # Update history for good points
        new_history = []
        history_idx = 0
        for i, good in enumerate(good_mask):
            if good and history_idx < len(self.point_history):
                self.point_history[history_idx].append(good_curr.reshape(-1, 2)[history_idx])
                new_history.append(self.point_history[history_idx])
                history_idx += 1
        
        self.point_history = new_history
        
        # Update state
        self.prev_gray = gray
        self.prev_points = good_curr.reshape(-1, 1, 2).astype(np.float32)
        
        return good_curr.reshape(-1, 2), displacements, len(good_curr)
    
    def compute_flutter_frequency(self) -> float:
        """
        ROBUST FREQUENCY ESTIMATION
        1. Analyzes Y-motion (vertical whipping) instead of X.
        2. Uses Zero-Padding for high-resolution output (fixes "staircase").
        3. Averages the spectra of ALL tracked points (noise reduction).
        """
        if len(self.point_history) == 0:
            return 0.0
        
        # Configuration
        MIN_HISTORY = 15      # Minimum frames to consider a point valid
        PAD_LENGTH = 1024     # Zero-pad to this length for smooth FFT
        FPS = self.target_fps
        
        # Accumulator for the "Average Spectrum"
        avg_spectrum = np.zeros(PAD_LENGTH // 2)
        count = 0
        
        for history in self.point_history:
            if len(history) < MIN_HISTORY:
                continue
                
            # Convert to numpy
            pts = np.array(list(history))
            
            # CRITICAL CHANGE 1: Use Y-axis (Vertical) motion
            # Flags whip UP and DOWN more reliably than they stretch Left/Right.
            signal_data = pts[:, 1] 
            
            # Detrend (Center around 0)
            signal_data = signal_data - np.mean(signal_data)
            
            # Windowing (Reduces spectral leakage)
            window = np.hanning(len(signal_data))
            windowed = signal_data * window
            
            # CRITICAL CHANGE 2: Zero-Padding (n=PAD_LENGTH)
            # This interpolates the FFT, turning "0.33 Hz steps" into a smooth curve.
            fft_vals = np.abs(np.fft.fft(windowed, n=PAD_LENGTH))
            
            # We only care about the first half (positive frequencies)
            half_spectrum = fft_vals[:PAD_LENGTH // 2]
            
            # Accumulate
            avg_spectrum += half_spectrum
            count += 1
            
        if count == 0:
            return 0.0
            
        # Normalize the average
        avg_spectrum /= count
        
        # Calculate Frequency Axis
        # The resolution is now FPS / PAD_LENGTH (e.g. 30 / 1024 = 0.029 Hz precision)
        freqs = np.fft.fftfreq(PAD_LENGTH, d=1.0/FPS)[:PAD_LENGTH // 2]
        
        # CRITICAL CHANGE 3: Smart Peak Finding
        # Ignore DC offset and very low freq wobble (0.5 Hz)
        # Ignore high frequency camera noise (> 12 Hz)
        valid_mask = (freqs > 0.5) & (freqs < 12.0)
        
        if not np.any(valid_mask):
            return 0.0
            
        # Extract valid range
        valid_freqs = freqs[valid_mask]
        valid_spectrum = avg_spectrum[valid_mask]
        
        # Find the dominant peak
        peak_idx = np.argmax(valid_spectrum)
        peak_freq = valid_freqs[peak_idx]
        peak_val = valid_spectrum[peak_idx]
        
        # Noise Gate: The peak must be distinct
        mean_noise = np.mean(valid_spectrum)
        if peak_val < mean_noise * 3.0:  # Signal must be 3x stronger than background
            return 0.0
            
        return float(peak_freq)
    
    def compute_extension_angle(self, points: np.ndarray) -> float:
        """
        Compute flag extension angle from horizontal.
        """
        if len(points) == 0 or self.pole_x is None:
            return 0.0
        
        # Centroid of tracked points
        centroid = np.mean(points, axis=0)
        
        # Horizontal distance from pole
        dx = centroid[0] - self.pole_x
        
        # Vertical offset (assume flag is horizontal at rest)
        # Use ROI center as reference
        if self.roi:
            roi_center_y = self.roi[1] + self.roi[3] / 2
            dy = centroid[1] - roi_center_y
        else:
            dy = 0
        
        # Angle from horizontal
        angle = np.degrees(np.arctan2(dy, abs(dx)))
        return float(angle)
    
    def apply_robust_filter(self, raw_value: float) -> float:
        """
        Robust output filter combining:
        1. Outlier rejection - ignore sudden jumps
        2. Rate limiting - clamp max change per frame
        3. EMA smoothing - smooth the output
        
        This handles tracking glitches (occlusion, lost points) gracefully.
        """
        # Step 1: Outlier rejection
        # If new value jumps too far from running average, use previous output
        if abs(raw_value - self.running_avg) > self.filter_outlier_thresh:
            filtered_input = self.filtered_wind  # Reject, use previous
        else:
            filtered_input = raw_value  # Accept
        
        # Step 2: EMA smoothing
        target = self.filter_alpha * filtered_input + (1 - self.filter_alpha) * self.filtered_wind
        
        # Step 3: Rate limiting
        delta = target - self.filtered_wind
        delta = max(-self.filter_max_rate, min(self.filter_max_rate, delta))
        self.filtered_wind = self.filtered_wind + delta
        
        # Update running average for outlier detection (slow update)
        self.running_avg = 0.05 * raw_value + 0.95 * self.running_avg
        
        return self.filtered_wind
    
    def compute_metrics(self, points: np.ndarray, displacements: np.ndarray) -> WindMetrics:
        """
        Compute all wind metrics from tracking data.
        """
        metrics = WindMetrics()
        
        if len(points) == 0:
            return metrics
        
        metrics.num_tracked_points = len(points)
        metrics.quality = min(1.0, len(points) / 10.0)
        
        # RMS displacement (instantaneous motion)
        if len(displacements) > 0:
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
            self.displacement_history.append(np.mean(displacement_magnitudes))
            
            if len(self.displacement_history) > 5:
                metrics.rms_displacement = float(
                    np.sqrt(np.mean(np.square(list(self.displacement_history))))
                )
        
        # Flutter frequency
        metrics.flutter_freq_hz = self.compute_flutter_frequency()
        
        # Extension angle
        metrics.extension_angle_deg = self.compute_extension_angle(points)
        
        # Wind speed estimation based on selected method
        if self.method == 'cubic_kalman':
            # Method 1: Cubic polynomial + Kalman filter
            raw_estimate = fit_cubic_model(metrics.rms_displacement, self.cubic_coeffs)
            
            # Apply Kalman filter
            x_pred = self.kalman_state
            P_pred = self.kalman_covariance + self.kalman_process_var
            K = P_pred / (P_pred + self.kalman_measurement_var)
            self.kalman_state = x_pred + K * (raw_estimate - x_pred)
            self.kalman_covariance = (1 - K) * P_pred
            
            metrics.estimated_wind_mps = max(0, self.kalman_state)
            
        elif self.method == 'log_kalman':
            # Method 2: Logarithmic model + Kalman filter
            raw_estimate = fit_logarithmic_model(metrics.rms_displacement, self.log_coeffs)
            
            # Apply Kalman filter
            x_pred = self.kalman_state
            P_pred = self.kalman_covariance + self.kalman_process_var
            K = P_pred / (P_pred + self.kalman_measurement_var)
            self.kalman_state = x_pred + K * (raw_estimate - x_pred)
            self.kalman_covariance = (1 - K) * P_pred
            
            metrics.estimated_wind_mps = max(0, self.kalman_state)
            
        elif self.method == 'robust_filter':
            # Method 3: Legacy linear fusion + robust filter
            v_rms = self.cal_rms_gain * metrics.rms_displacement + self.cal_rms_offset
            v_freq = self.cal_freq_gain * metrics.flutter_freq_hz
            v_angle = self.cal_angle_gain * abs(metrics.extension_angle_deg)
            
            raw_wind = max(0, (
                self.weight_rms * v_rms +
                self.weight_freq * v_freq +
                self.weight_angle * v_angle
            ))
            
            metrics.estimated_wind_mps = self.apply_robust_filter(raw_wind)
            
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'cubic_kalman', 'log_kalman', or 'robust_filter'")
        
        return metrics
    
    def read_ground_truth(self) -> bool:
        """
        Read ground truth from serial. Returns True if new value received.
        """
        if self.gt_serial is None:
            return False
        
        try:
            raw = self.gt_serial.readline().decode(errors='ignore').strip()
            if raw:
                match = re.search(r"[-+]?\d*\.?\d+", raw)
                if match:
                    self.last_gt = float(match.group(0))
                    return True
        except:
            pass
        
        return False
    
    def draw_debug(self, frame: np.ndarray, points: np.ndarray, 
                   metrics: WindMetrics) -> np.ndarray:
        """Draw debug visualization"""
        debug = frame.copy()
        
        # Draw ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw pole position
        if self.pole_x:
            cv2.line(debug, (self.pole_x, 0), (self.pole_x, frame.shape[0]), 
                    (255, 0, 255), 1)
        
        # Draw tracked points
        for pt in points:
            cv2.circle(debug, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
        
        # Draw point trails
        for history in self.point_history:
            if len(history) > 1:
                pts = np.array(list(history)).astype(int)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    color = (int(255*alpha), int(100*alpha), 0)
                    cv2.line(debug, tuple(pts[i-1]), tuple(pts[i]), color, 1)
        
        # Draw metrics
        y_offset = 30
        texts = [
            f"Points: {metrics.num_tracked_points} (Q:{metrics.quality:.1%})",
            f"RMS Disp: {metrics.rms_displacement:.2f} px",
            f"Flutter: {metrics.flutter_freq_hz:.2f} Hz",
            f"Angle: {metrics.extension_angle_deg:.1f} deg",
            f"Wind Est: {metrics.estimated_wind_mps:.2f} m/s",
            f"GT: {self.last_gt:.2f} m/s" if not np.isnan(self.last_gt) else "GT: N/A"
        ]
        
        for text in texts:
            cv2.putText(debug, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return debug
    
    def run(self, display: bool = True):
        """Main sensor loop"""
        print("\n" + "="*50)
        print("ROBUST WIND SENSOR")
        print("="*50)
        print(f"Method: {self.method}")
        if self.method == 'cubic_kalman':
            print("  Using: Cubic polynomial + Kalman filter (R²=0.8814)")
        elif self.method == 'log_kalman':
            print("  Using: Logarithmic model + Kalman filter (R²=0.8306)")
        elif self.method == 'robust_filter':
            print("  Using: Linear fusion + Robust filter")
        print("="*50)
        print("Controls:")
        print("  'r' - Reset ROI selection")
        print("  't' - Re-detect tracking points")
        print("  'q' - Quit")
        print("="*50 + "\n")
        
        frame_count = 0
        
        try:
            while True:
                frame = self.capture_frame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Initial ROI selection
                if self.roi is None:
                    self.select_roi(frame)
                    continue
                
                # Track points
                points, displacements, num_good = self.track_points(gray)
                
                # Compute metrics
                metrics = self.compute_metrics(points, displacements)
                
                # Read ground truth
                new_gt = self.read_ground_truth()
                
                # Log data
                ts = time.time()
                self.log_writer.writerow([
                    ts, metrics.rms_displacement, metrics.flutter_freq_hz,
                    metrics.extension_angle_deg, metrics.estimated_wind_mps,
                    metrics.num_tracked_points, metrics.quality, self.last_gt
                ])
                
                # Send over UART to MacBook
                self.send_uart(metrics)
                
                # Log calibration data when GT is available
                if new_gt:
                    self.cal_writer.writerow([
                        ts, metrics.rms_displacement, metrics.flutter_freq_hz,
                        metrics.extension_angle_deg, self.last_gt
                    ])
                    self.cal_file.flush()
                    print(f"[CAL] GT={self.last_gt:.2f} RMS={metrics.rms_displacement:.2f} "
                          f"Freq={metrics.flutter_freq_hz:.2f}")
                
                # Periodic log flush
                frame_count += 1
                if frame_count % 30 == 0:
                    self.log_file.flush()
                
                # Display
                if display:
                    debug = self.draw_debug(frame, points, metrics)
                    cv2.imshow("Wind Sensor", debug)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi = None
                    self.prev_points = None
                    self.prev_gray = None
                elif key == ord('t'):
                    self.prev_points = None
                    print("[INFO] Re-detecting tracking points...")
        
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.use_picamera:
            try:
                self.camera.stop()
            except:
                pass
        else:
            self.camera.release()
        
        if self.log_file:
            self.log_file.close()
        if self.cal_file:
            self.cal_file.close()
        if self.gt_serial:
            self.gt_serial.close()
        if self.uart_out:
            self.uart_out.close()
        
        cv2.destroyAllWindows()
        print("[OK] Cleanup complete")


def main():
    sensor = RobustWindSensor(use_picamera=True)
    
    # UART output to MacBook
    sensor.connect_uart_output("/dev/ttyAMA0", 115200)
    
    # Ground truth input (optional - comment out if GT is on MacBook only)
    # sensor.connect_ground_truth("/dev/ttyACM0")
    
    sensor.run(display=True)


if __name__ == "__main__":
    main()