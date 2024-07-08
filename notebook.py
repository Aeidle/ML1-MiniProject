import cv2
import pandas as pd
import os
from datetime import datetime
import csv
import time
import face_recognition
from scipy.spatial import distance
import dlib
from imutils import face_utils
import threading

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indices of facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_tiredness(frame, timestamp):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.25:
            print("Tiredness detected")
            save_ear(timestamp, ear, istired=1, csv_path=r"output\ear_values.csv")
        else:
            print("No tiredness detected")
            save_ear(timestamp, ear, istired=0, csv_path=r"output\ear_values.csv")

def process_frame(frame, csv_path):
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
    
    # Write the face count to the CSV file
    save_face_counts(timestamp, face_count, csv_path)
    
    # Calculate tiredness
    calculate_tiredness(rgb_frame, timestamp)
    
    # Display the frame
    cv2.imshow("Video Stream", frame)

def video_thread(width, height, video_path, csv_path, interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            # Resize the frame
            frame = cv2.resize(frame, (width, height))
            if frame_count % interval == 0:
                # Create a new thread for processing the current frame
                threading.Thread(target=process_frame, args=(frame, csv_path)).start()
                
            # Display the frame
            cv2.imshow("Video Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(width, height, video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps) * 1  # Extract frame every 1 second
    cap.release()
    
    # Start the video processing thread
    video_thread(width, height, video_path, csv_path, interval)

# Example usage
width = 640
height = 480
# video_path = r"src\vid1.mp4"
video_path = 0
csv_path = r"output\face_counts.csv"

main(width, height, video_path, csv_path)




# Example usage
# width = 640
# height = 480
# video_path = r"src\vid1.mp4"
# csv_path = r"output\face_counts.csv"

# process_video(width, height, video_path, csv_path)
