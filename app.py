from flask import Flask, render_template, Response
import cv2
import numpy as np
from base64 import b64encode

app = Flask(__name__)

# Function to perform face detection on an image and return the image with bounding boxes around the detected faces
def detect_faces(image):
    # Load the OpenCV face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Return the image with bounding boxes around the detected faces
    return image

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for streaming the video with face detection
@app.route('/video_feed')
def video_feed():
    # Open a connection to the user's camera
    cap = cv2.VideoCapture(0)

    # Define the MIME type for streaming video
    video_mime_type = 'multipart/x-mixed-replace; boundary=frame'

    def generate():
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Call the detect_faces function to perform face detection on the frame
            result_frame = detect_faces(frame)

            # Convert the result image to a JPEG-encoded byte string for streaming
            _, result_buffer = cv2.imencode('.jpg', result_frame)
            result_str = result_buffer.tobytes()

            # Send the result image as a MIME multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + result_str + b'\r\n')

    # Return the streaming response
    return Response(generate(), mimetype=video_mime_type)

if __name__ == '__main__':
    app.run(debug=True)
