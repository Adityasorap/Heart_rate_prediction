# Heart_rate_prediction
# Heart Rate Monitoring System

## Overview
This is a real-time heart rate monitoring system that uses a webcam to capture facial video and estimate the heart rate using the green channel intensity variations. It employs OpenCV for face detection and SciPy for peak detection in the signal data.

## Features
- Uses OpenCV's Haar Cascade for face detection.
- Extracts green channel intensity to measure heart rate.
- Uses SciPy's peak detection to calculate heart rate.
- Streams real-time video with heart rate overlay.
- Flask WebSocket (SocketIO) support for real-time updates.

## Requirements
Install the dependencies using the following command:
```bash
pip install flask opencv-python numpy scipy flask-socketio
```

## Running the Application
1. Run the Python script:
```bash
python app.py
```
2. Open a browser and go to:
```
http://127.0.0.1:5001/
```
3. The webpage will display the live video feed with real-time heart rate updates.

## API Endpoints
- `/`: Main webpage with video feed.
- `/video_feed`: Streams the video feed.
- `/start_monitoring`: Starts heart rate monitoring.
- `/stop_monitoring`: Stops monitoring and displays the last calculated heart rate.

## WebSocket Events
- `heart_rate`: Sends real-time heart rate updates to the frontend.
- `connect`: Notifies when a client connects.
- `disconnect`: Notifies when a client disconnects.

## Notes
- Ensure your webcam is properly connected.
- Works best in good lighting conditions.
- Face should be stable and not move excessively for accurate readings.

## License
This project is open-source and can be modified as needed.
