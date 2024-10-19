from flask import Flask, jsonify, request, Response, render_template, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import csv
import os
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLOv8-pose 모델 로드
model = YOLO('yolov8n-pose.pt')

# 각도 계산 함수
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ab = a - b
    cb = c - b
    
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# 비디오 처리 함수
def process_video(video_path, csv_filename):
    cap = cv2.VideoCapture(video_path)
    frame_number = 1

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Updated CSV header to include keypoints
        header = ['frame_number', 'keypoints', 'elbow_angle', 'knee_angle']
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임에서 포즈 추출
            results = model(frame)
            keypoints = results[0].keypoints.xyn.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            max_area = 0
            selected_person = None

            for person, box in zip(keypoints, boxes):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)

                if area > max_area:
                    max_area = area
                    selected_person = person

            if selected_person is not None and 94 <= frame_number <= 132:
                # Calculating the angles
                left_shoulder = selected_person[5][:2]
                left_elbow = selected_person[7][:2]
                left_wrist = selected_person[9][:2]
                elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                left_hip = selected_person[11][:2]
                left_knee = selected_person[13][:2]
                left_ankle = selected_person[15][:2]
                knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # Saving the keypoints as a list of tuples
                keypoints_list = [(kp[0], kp[1]) for kp in selected_person]

                # Writing the data into the CSV file
                writer.writerow([frame_number, keypoints_list, elbow_angle, knee_angle])

            frame_number += 1

    cap.release()

# 비디오 업로드 및 처리 라우트
@app.route('/upload', methods=['POST'])
def upload_video():
    datetime_now_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if 'video' not in request.files:
        return "No video file", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected video", 400
    else :
        video.filename = datetime_now_string+'.MOV'

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    csv_filename = 'output/{}.csv'.format(datetime_now_string)
    process_video(video_path, csv_filename)

    return send_file(csv_filename, as_attachment=True)

# 메인 페이지 라우트
@app.route('/')
def index():
    upload_video_name = os.listdir('./uploads')
    print(upload_video_name)
    return render_template('index.html', upload_video_name=upload_video_name)

if __name__=="__main__":
    app.run(debug=True, port=3000, host='0.0.0.0')
