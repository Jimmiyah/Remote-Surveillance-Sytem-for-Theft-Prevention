from flask import Flask, Response,render_template
from picamera2 import Picamera2
import cv2
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import time
# Initialize Flask app
app = Flask(__name__)

# Initialize PiCamera2
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

# Load the pre-trained MobileNet-SSD model
prototxt_path = "/home/jimitdesai/deploy.prototxt"
model_path = "/home/jimitdesai/mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# Object detection class labels (MobileNet-SSD)
class_labels = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
# Email settings
EMAIL_ADDRESS = "jimitdesai24@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "Admin@123456"  # Replace with your email password or app password
RECIPIENT_EMAIL = "jimitdesai24@gmail.com"  # Replace with recipient email
# Global variables for video stream and email
output_frame = None
lock = threading.Lock()
email_sent = False
last_email_time=0.0
# Function to send an email
def send_email():
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Person Detected"

        body = "A person has been detected in the camera feed."
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            text = msg.as_string()
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, text)

        print("Email sent successfully.")
        last_email_time=time.time()

    except Exception as e:
        print(f"Failed to send email: {e}")
# Function for video streaming and object detection
def generate_frames():
    """Video streaming with object detection."""
    global last_email_time

    while True:
        try:
            # Capture a frame from PiCamera2
            frame = picam2.capture_array()

            # Convert to BGR if needed (ensure 3 channels)
            if frame.shape[2] == 4:  # If RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 1:  # If grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Convert the frame to a blob for the DNN
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Iterate over detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                # Send email only if person is detected and 10 seconds have passed
                if confidence > 0.5 and class_labels[class_id] == "person":
                    current_time = time.time()
                    if current_time - last_email_time >= 10:  # 10-second buffer
                        send_email()
                        last_email_time = current_time  # Update last email time

                    # Draw bounding box and label for "person" detection
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                    )
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    text = f"{class_labels[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a byte stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error generating frame: {e}")
            break
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
