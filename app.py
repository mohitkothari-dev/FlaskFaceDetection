from flask import Flask, render_template, request
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

# Define the route for processing the uploaded image
@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded file from the request object
    file = request.files['image']

    # Read the image as a numpy array
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Call the detect_faces function to perform face detection on the image
    result_img = detect_faces(img)

    # Convert the result image to a JPEG-encoded byte string for display on the web page
    _, result_buffer = cv2.imencode('.jpg', result_img)
    result_str = result_buffer.tobytes()

    # Return the result image as a response
    return render_template('result.html', result_image=b64encode(result_str).decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True)
