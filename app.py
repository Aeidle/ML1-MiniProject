from flask import Flask, render_template, Response, request, redirect, url_for
from flask_socketio import SocketIO
import cv2
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import dlib
from imutils import face_utils
from scipy.spatial import distance
import face_recognition
import os
import sys
import torch
import torchvision.transforms as transforms
import joblib
from efficientnet_pytorch import EfficientNet
from PIL import Image
from flask import jsonify


app = Flask(__name__)
socketio = SocketIO(app)

# Load dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indices of facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Engagement
categories = ['writing_reading', 'distracted_mouth_open', 'using_smartphone', 'focused_mouth_closed', 
              'distracted_mouth_closed', 'fatigue', 'focused_mouth_open', 'raise_hand', 'listening', 'sleeping']

label_map = {'writing_reading': 0,
 'distracted_mouth_open': 1,
 'using_smartphone': 2,
 'focused_mouth_closed': 3,
 'distracted_mouth_closed': 4,
 'fatigue': 5,
 'focused_mouth_open': 6,
 'raise_hand': 7,
 'listening': 8,
 'sleeping': 9}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the feature extractor model
model = EfficientNet.from_pretrained('efficientnet-b0')
model.load_state_dict(torch.load('models/efficientnet_b0.pth'))
model.to(device)
model.eval()

# Load Classification model
classifier = joblib.load(r"models/Logistic Regression_model.pkl")

# load clustering model
kmeans_model = joblib.load(r"models\KMeans_model.pkl")

# Load PCA
pca = joblib.load(r"models\pca_model.pkl")


# Function to preprocess frame
def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    return img_tensor

# Store the last 20 records
face_data = []
ear_data = []
engagement_data = []
engagement_bar = []
prediction_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Function to process each frame
def process_frame_periodically(frame):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")
    
    # Convert the frame from BGR (OpenCV's default) to RGB (face_recognition's requirement)
    rgb_frame = frame[:, :, ::-1]

    # Use face_recognition to detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Draw bounding box around faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # Save the number of faces detected in the current frame
    face_count = len(face_locations)
    save_face_counts(timestamp, face_count, "output/face_counts.csv")

    # Calculate tiredness
    ear_value, istired = calculate_tiredness(rgb_frame, timestamp)

    # Preprocess the frame for prediction
    img_tensor = preprocess_frame(frame)

    # Perform the prediction
    with torch.no_grad():
        feature = model.extract_features(img_tensor).flatten().cpu().numpy()
    prediction = classifier.predict([feature])
    predicted_label = list(label_map.keys())[list(label_map.values()).index(prediction[0])]
    
    label = ' '.join([word.capitalize() for word in predicted_label.split("_")])
    
    # print(f"{prediction[0] = }")
    # print(f"{predicted_label = }")
    
    prediction_count[prediction[0]] += 1
    
    # print(prediction_count)
    
    save_engagement(timestamp, predicted_label, csv_path="output/engagements.csv")

    # Add to data lists
    face_data.append({'timestamp': timestamp, 'face_count': face_count})
    ear_data.append({'timestamp': timestamp, 'ear_value': istired})
    # engagement_data.append({'timestamp': timestamp, 'engagement': predicted_label})
    # engagement_bar.append({'classes': categories, 'engagement_count': prediction_count})
    
    # Ensure we only keep the last 20 records
    if len(face_data) > 50:
        face_data.pop(0)
    if len(ear_data) > 50:
        ear_data.pop(0)
    # if len(engagement_data) > 20:
    #     engagement_data.pop(0)
    # if len(engagement_bar) > 1:
    #     engagement_data.pop(0)

    # Send updated data to the client via WebSocket
    socketio.emit('update_data', {
        'face_data': face_data, 
        'ear_data': ear_data, 
        'label': label,
        'engagement_bar': prediction_count})


# Ensure output directory exists
os.makedirs('output', exist_ok=True)

def save_face_counts(timestamp, face_count, csv_path):
    with open(csv_path, "a", newline="\n") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['time', 'face_count'])
        writer.writerow([timestamp, face_count])

def save_ear(timestamp, ear_value, istired, csv_path):
    with open(csv_path, "a", newline="\n") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['time', 'ear_value', 'istired'])
        writer.writerow([timestamp, ear_value, istired])
        
def save_engagement(timestamp, engagement, csv_path):
    with open(csv_path, "a", newline="\n") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['time', 'engagement'])
        writer.writerow([timestamp, engagement])

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_tiredness(frame, timestamp):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    ear = None  # Initialize ear variable
    istired = 0

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.25:
            istired=1
            save_ear(timestamp, ear, istired=istired, csv_path="output/ear_values.csv")
        else:
            istired=0
            save_ear(timestamp, ear, istired=istired, csv_path="output/ear_values.csv")

    if ear is None:
        ear = 0.0  # or set to some default value if no subjects are found

    return ear, istired


def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to 0 for webcam or provide video path
    if not cap.isOpened():
        print("Failed to open video source")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from video source")
            break

        process_frame_periodically(frame)
        
        # Convert the frame from BGR (OpenCV's default) to RGB (face_recognition's requirement)
        rgb_frame = frame[:, :, ::-1]

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

@app.route('/perform_clustering', methods=['POST'])
def perform_clustering():
    try:
        all_clusters = [2, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 0, 0,
       1, 1, 2, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2,
       0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 2, 1, 1,
       0, 1, 2, 0, 1, 2, 1, 1, 0, 0, 1, 2, 1, 2, 0, 0, 0, 1, 0, 1, 1, 2,
       0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1,
       0, 1, 1, 0, 0, 0, 1, 2, 0, 2, 1, 1, 1, 0, 1, 2, 1, 2, 0, 2, 0, 0,
       2, 1, 2, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
       1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0,
       1, 2, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 1, 0, 2,
       2, 0, 1, 2, 0, 0, 2, 0, 2, 1, 2, 1, 0, 1, 0, 1, 2, 0, 2, 1, 0, 1,
       1, 0, 0, 0, 1, 2, 1, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0,
       1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 2,
       0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 1, 1, 0, 0,
       0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0,
       1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2,
       2, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 1, 0, 2, 2, 0, 0,
       1, 0, 0, 2, 0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1,
       2, 1, 1, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0,
       2, 1, 1, 0, 0, 0, 0, 1, 2, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 2, 1, 0,
       0, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 0, 1, 0, 0,
       2, 1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
       0, 1, 0, 1, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 1, 2, 1, 1, 1, 0,
       1, 0]
        
        cluster_centers_ = [[ 172.68054857,    8.0268107 ,   -5.83040812],
       [-141.10697993,  -68.3705713 ,    3.78841119],
       [-153.99792139,  125.29465308,    7.29091178]]
        
        X_original = np.load(r'models\X_data.npy')
        
        group = ''
        # Get form data
        data = {
            '# Logins': request.form['logins'],
            '# Content Reads': request.form['content_reads'],
            '# Forum Reads': request.form['forum_reads'],
            '# Forum Posts': request.form['forum_posts'],
            '# Quiz Reviews before submission': request.form['quiz_reviews'],
            'Assignment 1 lateness indicator': request.form['assignment_1_lateness'],
            'Assignment 2 lateness indicator': request.form['assignment_2_lateness'],
            'Assignment 3 lateness indicator': request.form['assignment_3_lateness'],
            'Assignment 1 duration to submit (in hours)': request.form['assignment_1_duration'],
            'Assignment 2 duration to submit (in hours)': request.form['assignment_2_duration'],
            'Assignment 3 duration to submit (in hours)': request.form['assignment_3_duration'],
            'Average time to submit assignment (in hours)': request.form['average_time_to_submit']
        }
        
        # Convert data to pandas DataFrame
        new_data = pd.DataFrame(data, index=[0])
        
        # print(new_data.head(1))
        
        # Transform the new data using PCA
        pca_data = pca.transform(new_data)
        
        X = pca_data[:, 0:3]
        
        # Predict the cluster for the new data using the pre-trained KMeans model
        y_kmeans = kmeans_model.predict(X)
        
        group = y_kmeans[0] + 1
        
        print(f"{group = }")
        
        
        
        # send_cluster_result(group)
        
        # return redirect(url_for('clustering'))
        return jsonify({
            'group': str(group),
            'cluster_centers': cluster_centers_,
            'data': all_clusters,
            'X': X.tolist(),
            'X_original': X_original.tolist()
            })
    except Exception as e:
        print(f"Error during clustering: {e}")
        return "Error during clustering", 500


if __name__ == "__main__":
    # Start the Flask app with SocketIO
    socketio.run(app, debug=True)
