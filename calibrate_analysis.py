#!/usr/bin/env python3
"""
Wind Sensor Calibration Analysis
================================

Run this after collecting wind tunnel data to:
1. Analyze correlation between metrics and ground truth
2. Fit calibration coefficients
3. Generate calibration report and plots
"""

import numpy as np
import csv
import sys
from pathlib import Path


def load_calibration_data(filename: str) -> dict:
    """Load calibration CSV data"""
    data = {
        'timestamp': [],
        'rms_displacement': [],
        'flutter_freq_hz': [],
        'extension_angle_deg': [],
        'gt_wind_mps': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data.keys():
                try:
                    data[key].append(float(row[key]))
                except (ValueError, KeyError):
                    data[key].append(np.nan)
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Remove NaN rows
    valid = ~np.isnan(data['gt_wind_mps'])
    for key in data:
        data[key] = data[key][valid]
    
    return data


def compute_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient"""
    if len(x) < 3:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple:
    """Fit y = gain * x + offset, return (gain, offset, r_squared)"""
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    
    # Linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    gain, offset = result[0]
    
    # R-squared
    y_pred = gain * x + offset
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return float(gain), float(offset), float(r_squared)


def analyze_calibration(filename: str):
    """Main calibration analysis"""
    print("=" * 60)
    print("WIND SENSOR CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading: {filename}")
    data = load_calibration_data(filename)
    n_samples = len(data['gt_wind_mps'])
    print(f"Samples: {n_samples}")
    
    if n_samples < 10:
        print("\n[ERROR] Need at least 10 calibration samples!")
        print("Run the sensor in the wind tunnel with ground truth connected.")
        return
    
    gt = data['gt_wind_mps']
    print(f"\nGround Truth Range: {gt.min():.2f} - {gt.max():.2f} m/s")
    print(f"Ground Truth Mean:  {gt.mean():.2f} m/s")
    
    # Analyze each metric
    print("\n" + "-" * 60)
    print("INDIVIDUAL METRIC ANALYSIS")
    print("-" * 60)
    
    metrics = [
        ('rms_displacement', 'RMS Displacement (px)'),
        ('flutter_freq_hz', 'Flutter Frequency (Hz)'),
        ('extension_angle_deg', 'Extension Angle (deg)')
    ]
    
    fits = {}
    
    for key, name in metrics:
        x = data[key]
        
        # For angle, use absolute value
        if 'angle' in key:
            x = np.abs(x)
        
        corr = compute_correlation(x, gt)
        gain, offset, r2 = fit_linear(x, gt)
        
        fits[key] = {'gain': gain, 'offset': offset, 'r2': r2, 'corr': corr}
        
        print(f"\n{name}:")
        print(f"  Range:       {x.min():.3f} - {x.max():.3f}")
        print(f"  Correlation: {corr:.3f}")
        print(f"  R-squared:   {r2:.3f}")
        print(f"  Fit: wind = {gain:.4f} * x + {offset:.4f}")
    
    # Determine best single metric
    best_metric = max(fits.keys(), key=lambda k: fits[k]['r2'])
    print(f"\n>>> Best single metric: {best_metric} (R² = {fits[best_metric]['r2']:.3f})")
    
    # Multi-metric fusion
    print("\n" + "-" * 60)
    print("MULTI-METRIC FUSION")
    print("-" * 60)
    
    # Build feature matrix
    X = np.column_stack([
        data['rms_displacement'],
        data['flutter_freq_hz'],
        np.abs(data['extension_angle_deg']),
        np.ones(n_samples)  # bias term
    ])
    
    # Least squares fit
    result = np.linalg.lstsq(X, gt, rcond=None)
    weights = result[0]
    
    w_rms, w_freq, w_angle, bias = weights
    
    # Compute R-squared for fusion
    y_pred = X @ weights
    ss_res = np.sum((gt - y_pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2_fusion = 1 - (ss_res / ss_tot)
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((gt - y_pred) ** 2))
    
    print(f"\nFused Model:")
    print(f"  wind = {w_rms:.4f} * rms + {w_freq:.4f} * freq + {w_angle:.4f} * |angle| + {bias:.4f}")
    print(f"\n  R-squared: {r2_fusion:.3f}")
    print(f"  RMSE:      {rmse:.3f} m/s")
    
    # Normalize weights for the sensor config
    total_weight = abs(w_rms) + abs(w_freq) + abs(w_angle)
    if total_weight > 0:
        norm_w_rms = abs(w_rms) / total_weight
        norm_w_freq = abs(w_freq) / total_weight
        norm_w_angle = abs(w_angle) / total_weight
    else:
        norm_w_rms, norm_w_freq, norm_w_angle = 0.33, 0.33, 0.34
    
    # Generate config
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    print(f"""
# Copy these values to wind_sensor_robust.py:

# Calibration gains (metric → m/s contribution)
self.cal_rms_gain = {w_rms:.4f}
self.cal_rms_offset = 0.0  # Absorbed into fusion bias
self.cal_freq_gain = {w_freq:.4f}
self.cal_angle_gain = {w_angle:.4f}

# Fusion weights (normalized)
self.weight_rms = {norm_w_rms:.3f}
self.weight_freq = {norm_w_freq:.3f}
self.weight_angle = {norm_w_angle:.3f}

# Or use simpler single-metric approach:
# Best metric: {best_metric}
# wind_mps = {fits[best_metric]['gain']:.4f} * {best_metric} + {fits[best_metric]['offset']:.4f}
""")
    
    # Error analysis by wind speed range
    print("-" * 60)
    print("ERROR ANALYSIS BY WIND SPEED")
    print("-" * 60)
    
    # Bin data by ground truth
    bins = [0, 2, 4, 6, 8, 10, 15, 20, np.inf]
    
    for i in range(len(bins) - 1):
        mask = (gt >= bins[i]) & (gt < bins[i+1])
        if np.sum(mask) > 0:
            bin_gt = gt[mask]
            bin_pred = y_pred[mask]
            bin_rmse = np.sqrt(np.mean((bin_gt - bin_pred) ** 2))
            bin_bias = np.mean(bin_pred - bin_gt)
            print(f"  {bins[i]:4.0f} - {bins[i+1]:4.0f} m/s: "
                  f"n={np.sum(mask):3d}, RMSE={bin_rmse:.2f}, bias={bin_bias:+.2f}")
    
    # Try to generate plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: RMS vs GT
        ax = axes[0, 0]
        ax.scatter(data['rms_displacement'], gt, alpha=0.5, s=10)
        x_fit = np.linspace(0, data['rms_displacement'].max(), 100)
        ax.plot(x_fit, fits['rms_displacement']['gain'] * x_fit + 
                fits['rms_displacement']['offset'], 'r-', linewidth=2)
        ax.set_xlabel('RMS Displacement (px)')
        ax.set_ylabel('Ground Truth (m/s)')
        ax.set_title(f"RMS vs Wind (R²={fits['rms_displacement']['r2']:.3f})")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Frequency vs GT
        ax = axes[0, 1]
        ax.scatter(data['flutter_freq_hz'], gt, alpha=0.5, s=10)
        x_fit = np.linspace(0, data['flutter_freq_hz'].max(), 100)
        ax.plot(x_fit, fits['flutter_freq_hz']['gain'] * x_fit + 
                fits['flutter_freq_hz']['offset'], 'r-', linewidth=2)
        ax.set_xlabel('Flutter Frequency (Hz)')
        ax.set_ylabel('Ground Truth (m/s)')
        ax.set_title(f"Frequency vs Wind (R²={fits['flutter_freq_hz']['r2']:.3f})")
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Predicted vs Actual
        ax = axes[1, 0]
        ax.scatter(gt, y_pred, alpha=0.5, s=10)
        ax.plot([0, gt.max()], [0, gt.max()], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Ground Truth (m/s)')
        ax.set_ylabel('Predicted (m/s)')
        ax.set_title(f"Fused Model (R²={r2_fusion:.3f}, RMSE={rmse:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Time series
        ax = axes[1, 1]
        t = data['timestamp'] - data['timestamp'][0]
        ax.plot(t, gt, 'b-', label='Ground Truth', alpha=0.7)
        ax.plot(t, y_pred, 'r-', label='Predicted', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wind Speed (m/s)')
        ax.set_title('Time Series Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = filename.replace('.csv', '_analysis.png')
        plt.savefig(plot_filename, dpi=150)
        print(f"\n[OK] Plot saved: {plot_filename}")
        plt.show()
        
    except ImportError:
        print("\n[INFO] Install matplotlib for plots: pip install matplotlib")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cal_file = sys.argv[1]
    else:
        # Look for most recent calibration file
        cal_files = list(Path('.').glob('calibration_*.csv'))
        if cal_files:
            cal_file = str(max(cal_files, key=lambda p: p.stat().st_mtime))
            print(f"Using most recent: {cal_file}")
        else:
            print("Usage: python calibrate_analysis.py <calibration_file.csv>")
            print("No calibration files found in current directory.")
            sys.exit(1)
    
    analyze_calibration(cal_file)