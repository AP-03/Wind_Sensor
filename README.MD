# Flag Design Guide for Optical Wind Sensing

### Design: Dot Array (High Contrast Points)
```
┌────────────────────────────────────┐
│                                    │
│  [POLE]   ○   ○   ○   ●   ●   ●   │  ← Flag tip (wind direction →)
│           ○   ○   ○   ●   ●   ●   │
│                                    │
└────────────────────────────────────┘

○ = White dots on black strip
● = Black dots on white strip
```

**Specs:**
- **Dot diameter:** 8-12mm (should appear ~10-20 pixels in camera)
- **Spacing:** 15-20mm between dots
- **Pattern:** 2 rows × 3-6 columns near flag tip
- **Colors:** Black dots on white background (highest contrast)

**Why it works:**
- Each dot is independently trackable
- Redundancy: lose some dots, still get measurements
- Clear centroid detection


## Flag Dimensions

For a 1280×720 camera at ~50cm distance:

```
Recommended flag size:
┌─────────────────────────────────┐
│                                 │
│         20cm wide               │
├─────────────────────────────────┤
│                                 │  8cm
│                                 │  tall
└─────────────────────────────────┘

Marker zone (last 5cm):
                    ├───────────┤
                    │ ●  ●  ●   │  ← Tracking dots here
                    │ ●  ●  ●   │
                    ├───────────┤
```

### Camera Setup:
- Position camera **perpendicular** to flag plane
- Flag should fill **30-50%** of frame width
- Ensure **consistent lighting** (avoid direct sunlight)
- Lock **exposure and white balance** (the code does this)

### Calibration Protocol:
1. Start at 0 m/s, record 30 seconds
2. Increase wind speed in 0.5 m/s steps
3. Hold each speed for 30 seconds
4. Cover full range (0 to max wind tunnel speed)
5. Run `calibrate_analysis.py` on the data


## Quick Start Checklist

- [ ] White card stock flag (160-200gsm)
- [ ] 6 black dots (8-10mm) near tip in 2×3 pattern
- [ ] Dots spaced 15mm apart
- [ ] Camera ~50cm from flag, perpendicular view
- [ ] Ground truth sensor connected
- [ ] Run `wind_sensor.py`
- [ ] Select ROI around the dot pattern
- [ ] Click pole position
- [ ] Collect calibration data across wind speed range
- [ ] Run `calibrate_analysis.py` to fit coefficients