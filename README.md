# Sensor Calibration
Python project for calibrating RealSense RGB-D cameras and IMU sensors.

## Overview
This project automatically calibrates the intrinsic parameters of Intel RealSense cameras and the extrinsic parameters of IMU sensors using a checkerboard pattern.

## Features
- **Camera Intrinsics Calibration**: Estimation of RGB and Depth camera internal parameters
- **IMU Extrinsics Calibration**: Estimation of Gyroscope and Accelerometer transformation matrix to camera coordinate system
- **Real-time Visualization**: Real-time verification of checkerboard detection results
- **Comprehensive Logging**: Automatic saving of IMU sensor data (gyro/accel)

## Requirements
```bash
pip install opencv-python numpy pyrealsense2 pyyaml
```

## Hardware Requirements
- Intel RealSense RGB-D Camera (D435i, D455)
- Checkerboard pattern (default: 6x8, 25mm squares)

## Configuration
`camera_calibration.py` settings:
```python
TARGET_SHOTS = 500          # Target number of capture frames
CHECKER_ROWS = 6            # Number of checkerboard rows
CHECKER_COLS = 8            # Number of checkerboard columns
SQUARE_SIZE_M = 0.025       # Checkerboard square size (m)
ALIGN_TO_COLOR = True       # Align Depth to Color
```

`IMU_calibration.py` settings:
```python
GAP = 3.0                   # Maximum frame interval (seconds)
MIN_ANGLE = 10.0            # Minimum rotation angle (degrees)
gyro_thresh = 0.1           # Static interval threshold
```

## Algorithm Details

### Camera Calibration

- **Corner Detection**: `cv.findChessboardCorners` + sub-pixel refinement
- **Optimization**: `cv.calibrateCamera` with RATIONAL_MODEL flag
- **Metric**: Reprojection error (RMS)

### IMU Calibration

- **Rotation Estimation**: Gyroscope integration + PnP-based camera rotation
- **Gravity Alignment**: Static interval detection + accelerometer gravity vector
- **Optimization**: Kabsch algorithm (SVD-based rigid transformation)
