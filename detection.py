import cv2
import numpy as np
import tempfile
from inference_sdk import InferenceHTTPClient
import serial
import time
from flask import Flask, jsonify, Response
from flask_cors import CORS, cross_origin
import threading
import signal
import sys

# Initialize Flask app
app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = 'Content-Type'

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="EZS0CWphtKuyN68Ys1ts"
)

PROJECT_ID = "cigarette-butt-detector-0jv4z"
MODEL_VERSION = "5"
MODEL_ID = f"{PROJECT_ID}/{MODEL_VERSION}"

# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Initialize serial communication with Arduino
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

performance_metrics = {
    "cigarettesPicked": 0,
    "distanceCovered": 0.0,
    "battery": 90  # Example initial value
}

@app.route('/api/robotData', methods=['GET'])
@cross_origin()
def get_robot_data():
    return jsonify(performance_metrics)

def gen_frames():
    """Generate video frames for streaming with detection boxes."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform cigarette detection
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            cv2.imwrite(temp_image_file.name, frame)
            image_path = temp_image_file.name

        result = CLIENT.infer(image_path, model_id=MODEL_ID)

        if 'predictions' in result:
            for prediction in result['predictions']:
                x0 = int(prediction['x'] - prediction['width'] / 2)
                y0 = int(prediction['y'] - prediction['height'] / 2)
                x1 = int(prediction['x'] + prediction['width'] / 2)
                y1 = int(prediction['y'] + prediction['height'] / 2)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, "Cigarette", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Use yield to generate video frames as a byte array
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
@cross_origin()
def video_feed():
    # Video streaming route, gen_frames is the stream generator function
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def flask_thread():
    """Run the Flask server."""
    app.run(host='0.0.0.0', port=3001, use_reloader=False)

def detection_and_serial_thread():
    """Handle detection and serial communication."""
    global performance_metrics

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform cigarette detection
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            cv2.imwrite(temp_image_file.name, frame)
            image_path = temp_image_file.name

        result = CLIENT.infer(image_path, model_id=MODEL_ID)

        if 'predictions' are in result:
            for prediction in result['predictions']:
                x0 = int(prediction['x'] - prediction['width'] / 2)
                y0 = int(prediction['y'] - prediction['height'] / 2)
                x1 = int(prediction['x'] + prediction['width'] / 2)
                y1 = int(prediction['y'] + prediction['height'] / 2)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, "Cigarette", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print("go go")
                arduino.write(b'F')  # Command to Arduino to move forward
                performance_metrics["cigarettesPicked"] += 1  # Update cigarette count
                print("Cigarette detected! Moving forward...")
            arduino.write(b'S')  # Command to Arduino to stop

        else:
            print(f"Error in response: {result}")

        if arduino.in_waiting > 0:
            data = arduino.readline().decode('utf-8').strip()
            if ':' in data:
                key, value = data.split(':', 1)
                if key in performance_metrics:
                    try:
                        performance_metrics[key] = float(value)
                    except ValueError:
                        print(f"Warning: Unable to convert value to float: {value}")
            else:
                print(f"Unexpected data received from Arduino: {data}")

        cv2.imshow('Cigarette Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    arduino.close()

def cleanup(signum, frame):
    print('Cleaning up...')
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup)  # Handle termination signals

if __name__ == '__main__':
    # Create and start threads for Flask and detection
    flask_thread = threading.Thread(target=flask_thread)
    detection_thread = threading.Thread(target=detection_and_serial_thread)

    flask_thread.start()
    detection_thread.start()

    # Wait for both threads to complete
    flask_thread.join()
    detection_thread.join()
