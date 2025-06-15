# S1_STUDIO
The original code file for the algorithm "Was Here" requires configuring the pytorch environment and installing necessary library files. Please refer to the readme document for details

# YOLO Stream Project

## Overview
This project uses YOLOv8 for real-time person detection and silhouette extraction from camera feeds. It processes video from two USB cameras, extracts person silhouettes, and streams the results via NDI (Network Device Interface) and OSC (Open Sound Control) protocols.

## Features
- Real-time person detection using YOLOv8 segmentation model
- Silhouette extraction from detected persons
- NDI streaming of both original camera feed and silhouette
- OSC messaging for person count and contour data
- Automatic camera reconnection on failure
- Debug mode for visual monitoring

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NDIlib
- python-osc
- NDI Runtime installed on the system

## Hardware Setup
- Two USB cameras (e.g., Logitech C270)
- Camera 1: Used for person detection, silhouette extraction, and contour data
- Camera 2: Used for top-view real-time video streaming

## Configuration
- Default camera IDs: 5 for Camera 1, 6 for Camera 2
- OSC sends to localhost (127.0.0.1) on ports 9001 (count) and 9002 (contour)
- NDI streams are named 'PythonYOLO_NDI_Cam1_Silhouette' and 'PythonYOLO_NDI_Cam2_Original'

## Usage
1. Ensure NDI Runtime is installed
2. Connect both USB cameras
3. Run the script: `python silhouette_stream.py`
4. Press 'q' to quit
5. Press 'd' to toggle debug mode (shows camera views)

## Notes
- The script automatically attempts to reconnect Camera 2 if it fails
- Debug frames are saved to the 'debug_frames' directory
- YOLOv8 is configured to only detect persons (class 0) with confidence > 0.25
