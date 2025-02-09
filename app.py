from flask import Flask, render_template, Response
import cv2
import numpy as np
import scipy.signal as signal
import time
from flask_socketio import SocketIO, emit

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize OpenCV and Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
color_signal = []  # To store the color intensity values for heart rate calculation
frame_count = 30  # Number of frames to capture for one heart rate calculation
fps = 30  # Frames per second
current_heart_rate = None
running = True  # Indicates if the monitoring is ongoing

# Function to calculate heart rate (BPM)
def calculate_heart_rate(signal_data, fps):
    peaks, _ = signal.find_peaks(signal_data, distance=fps * 0.6)  # Minimum 0.6 seconds between peaks
    if len(peaks) > 1:
        times_between_peaks = np.diff(peaks) / fps  # Time between peaks in seconds
        avg_time_between_peaks = np.mean(times_between_peaks)  # Average time between peaks
        heart_rate = 60 / avg_time_between_peaks  # Heart rate in beats per minute
        return heart_rate
    else:
        return None

# Function to run the video stream
def video_stream():
    global color_signal, running, current_heart_rate

    cap = cv2.VideoCapture(0)
    global fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:  # Fallback to 30 FPS if detection fails
        fps = 30

    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw stable spots on face if faces are detected
        for (x, y, w, h) in faces:
            cv2.circle(frame, (x + int(w * 0.3), y + int(h * 0.35)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x + int(w * 0.7), y + int(h * 0.35)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x + int(w * 0.5), y + int(h * 0.55)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x + int(w * 0.5), y + int(h * 0.85)), 5, (0, 255, 0), -1)

            # Green channel signal from face region
            green_channel = frame[y:y + h, x:x + w, 1]
            mean_green_intensity = np.mean(green_channel)
            color_signal.append(mean_green_intensity)

        # Calculate heart rate after collecting enough frames
        if len(color_signal) >= frame_count:
            heart_rate = calculate_heart_rate(np.array(color_signal), fps)
            if heart_rate:
                current_heart_rate = heart_rate
            else:
                current_heart_rate = "Calculation Failed"
            color_signal = []  # Reset after calculation

        # Emit heart rate updates to the front-end via WebSocket
        socketio.emit('heart_rate', {'heart_rate': current_heart_rate})

        # Convert frame to JPEG for streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

# Route to show the video feed on the webpage
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start monitoring
@app.route('/start_monitoring')
def start_monitoring():
    global running
    running = True  # Set monitoring to True
    return "Monitoring Started"

# Route to stop monitoring
@app.route('/stop_monitoring')
def stop_monitoring():
    global running
    running = False  # Stop the monitoring
    return f"Your heart rate is {current_heart_rate} BPM"  # Display the calculated heart rate

# WebSocket event handler to update heart rate in real-time
@socketio.on('connect')
def on_connect():
    print('Client connected')

@socketio.on('disconnect')
def on_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app, debug=True, host='127.0.0.1', port=5001)
