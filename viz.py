#!/usr/bin/env python3
"""
Wind Sensor Data Analysis & Plotting
=====================================

Analyzes and plots data from wind_sensor_robust.py
Run: python plot_sensor_data.py <csv_file>
"""

import numpy as np
import csv
import sys
from pathlib import Path
from datetime import datetime


def load_sensor_data(filename: str) -> dict:
    """Load sensor CSV data"""
    data = {
        'timestamp': [],
        'rms_displacement': [],
        'flutter_freq_hz': [],
        'extension_angle_deg': [],
        'wind_estimate_mps': [],
        'num_points': [],
        'quality': []
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
    
    return data


def print_statistics(data: dict):
    """Print summary statistics"""
    print("=" * 60)
    print("SENSOR DATA STATISTICS")
    print("=" * 60)
    
    # Time info
    t = data['timestamp']
    duration = t[-1] - t[0] if len(t) > 1 else 0
    fps = len(t) / duration if duration > 0 else 0
    
    print(f"\nRecording Duration: {duration:.1f} seconds")
    print(f"Total Frames:       {len(t)}")
    print(f"Average FPS:        {fps:.1f}")
    
    # Metrics
    metrics = [
        ('rms_displacement', 'RMS Displacement (px)'),
        ('flutter_freq_hz', 'Flutter Frequency (Hz)'),
        ('extension_angle_deg', 'Extension Angle (deg)'),
        ('wind_estimate_mps', 'Wind Estimate (m/s)'),
        ('num_points', 'Tracked Points'),
        ('quality', 'Quality')
    ]
    
    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 60)
    
    for key, name in metrics:
        arr = data[key]
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            print(f"{name:<25} {valid.min():>10.3f} {valid.max():>10.3f} "
                  f"{valid.mean():>10.3f} {valid.std():>10.3f}")
    
    # Tracking quality analysis
    print("\n" + "-" * 60)
    print("TRACKING QUALITY ANALYSIS")
    print("-" * 60)
    
    quality = data['quality']
    num_points = data['num_points']
    
    good_tracking = np.sum(quality > 0.5) / len(quality) * 100
    excellent_tracking = np.sum(quality > 0.8) / len(quality) * 100
    lost_tracking = np.sum(num_points < 3) / len(num_points) * 100
    
    print(f"Good tracking (>50%):      {good_tracking:.1f}% of frames")
    print(f"Excellent tracking (>80%): {excellent_tracking:.1f}% of frames")
    print(f"Lost tracking (<3 points): {lost_tracking:.1f}% of frames")


def analyze_data(filename: str):
    """Main analysis function"""
    print(f"\nLoading: {filename}")
    data = load_sensor_data(filename)
    
    if len(data['timestamp']) < 2:
        print("[ERROR] Not enough data points!")
        return
    
    print_statistics(data)
    
    # Try to plot
    try:
        import matplotlib.pyplot as plt
        
        # Normalize time to start at 0
        t = data['timestamp'] - data['timestamp'][0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Wind Sensor Data: {Path(filename).name}', fontsize=14)
        
        # Plot 1: Wind Estimate
        ax = axes[0, 0]
        ax.plot(t, data['wind_estimate_mps'], 'b-', linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wind Speed (m/s)')
        ax.set_title('Estimated Wind Speed')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Plot 2: RMS Displacement
        ax = axes[0, 1]
        ax.plot(t, data['rms_displacement'], 'g-', linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RMS Displacement (px)')
        ax.set_title('RMS Displacement')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Plot 3: Flutter Frequency
        ax = axes[1, 0]
        ax.plot(t, data['flutter_freq_hz'], 'r-', linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Flutter Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Plot 4: Extension Angle
        ax = axes[1, 1]
        ax.plot(t, data['extension_angle_deg'], 'm-', linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (deg)')
        ax.set_title('Extension Angle')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Tracking Quality
        ax = axes[2, 0]
        ax.plot(t, data['quality'] * 100, 'c-', linewidth=0.8, alpha=0.8)
        ax.axhline(y=50, color='orange', linestyle='--', label='Good threshold')
        ax.axhline(y=80, color='green', linestyle='--', label='Excellent threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Quality (%)')
        ax.set_title('Tracking Quality')
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Number of Tracked Points
        ax = axes[2, 1]
        ax.plot(t, data['num_points'], 'k-', linewidth=0.8, alpha=0.8)
        ax.axhline(y=3, color='red', linestyle='--', label='Minimum (3)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Points')
        ax.set_title('Tracked Points')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = filename.replace('.csv', '_plots.png')
        plt.savefig(plot_filename, dpi=150)
        print(f"\n[OK] Plots saved: {plot_filename}")
        
        # Show correlation matrix
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation: RMS vs Wind Estimate
        ax = axes2[0]
        ax.scatter(data['rms_displacement'], data['wind_estimate_mps'], 
                   alpha=0.3, s=5, c=t, cmap='viridis')
        ax.set_xlabel('RMS Displacement (px)')
        ax.set_ylabel('Wind Estimate (m/s)')
        ax.set_title('RMS vs Wind Estimate')
        ax.grid(True, alpha=0.3)
        
        # Correlation: Frequency vs Wind Estimate
        ax = axes2[1]
        scatter = ax.scatter(data['flutter_freq_hz'], data['wind_estimate_mps'], 
                            alpha=0.3, s=5, c=t, cmap='viridis')
        ax.set_xlabel('Flutter Frequency (Hz)')
        ax.set_ylabel('Wind Estimate (m/s)')
        ax.set_title('Frequency vs Wind Estimate')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time (s)')
        
        plt.tight_layout()
        
        corr_filename = filename.replace('.csv', '_correlations.png')
        plt.savefig(corr_filename, dpi=150)
        print(f"[OK] Correlations saved: {corr_filename}")
        
        plt.show()
        
    except ImportError:
        print("\n[INFO] Install matplotlib for plots: pip install matplotlib")
        print("       Statistics shown above are still valid.")


def plot_realtime_style(filename: str, window_sec: float = 30.0):
    """
    Plot data in a rolling window style (useful for long recordings)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        data = load_sensor_data(filename)
        t = data['timestamp'] - data['timestamp'][0]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # Wind speed
        ax1 = axes[0]
        line1, = ax1.plot([], [], 'b-', linewidth=1)
        ax1.set_ylabel('Wind (m/s)')
        ax1.set_title('Wind Speed Estimate')
        ax1.grid(True, alpha=0.3)
        
        # RMS
        ax2 = axes[1]
        line2, = ax2.plot([], [], 'g-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMS (px)')
        ax2.set_title('RMS Displacement')
        ax2.grid(True, alpha=0.3)
        
        def init():
            ax1.set_xlim(0, window_sec)
            ax1.set_ylim(0, max(data['wind_estimate_mps'].max() * 1.1, 1))
            ax2.set_xlim(0, window_sec)
            ax2.set_ylim(0, max(data['rms_displacement'].max() * 1.1, 1))
            return line1, line2
        
        def update(frame):
            # Sliding window
            t_end = min(frame * 0.5, t[-1])
            t_start = max(0, t_end - window_sec)
            
            mask = (t >= t_start) & (t <= t_end)
            t_window = t[mask] - t_start
            
            line1.set_data(t_window, data['wind_estimate_mps'][mask])
            line2.set_data(t_window, data['rms_displacement'][mask])
            
            return line1, line2
        
        frames = int(t[-1] / 0.5) + 1
        anim = FuncAnimation(fig, update, frames=frames, init_func=init, 
                            blit=True, interval=50)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("[ERROR] matplotlib required for plotting")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for most recent sensor log
        patterns = ['wind_sensor_*.csv', 'wind_log*.csv']
        found_files = []
        for pattern in patterns:
            found_files.extend(Path('.').glob(pattern))
        
        if found_files:
            csv_file = str(max(found_files, key=lambda p: p.stat().st_mtime))
            print(f"Using most recent: {csv_file}")
        else:
            print("Usage: python plot_sensor_data.py <sensor_data.csv>")
            print("\nNo sensor log files found in current directory.")
            print("Looking for: wind_sensor_*.csv or wind_log*.csv")
            sys.exit(1)
    
    analyze_data(csv_file)