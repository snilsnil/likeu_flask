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
    
    # 허용 오차 설정
    height_tolerance = 0.01
    
    # 슛 던지는 시작 프레임
    start_frame=0

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
            results = model.track(source=frame, persist=True)  # persist=True로 ID 유지
            
            for result in results:
                boxes = result.boxes  # 각 추적된 객체의 바운딩 박스 정보
                keypoints = result.keypoints  # 각 객체의 keypoints 정보
                
                for i, box in enumerate(boxes):
                    track_id = int(box.id) if box.id is not None else -1  # 추적 ID
                    
                    if track_id == 1:  # ID가 1인 객체 필터링
                        keypoint=keypoints.xyn.cpu().numpy()  # 키포인트 좌표 
                        
                        # 관절 높이 가져오기
                        left_shoulder_y = keypoint[0][6][1]  # 왼쪽 어깨의 y 좌표
                        left_elbow_y = keypoint[0][8][1]     # 왼쪽 팔꿈치의 y 좌표
                        
                        # 팔꿈치가 어깨 높이와 동일할 때
                        if abs(left_elbow_y - left_shoulder_y) <= height_tolerance and start_frame==0:
                            start_frame=frame_number
                        
                        if start_frame <= frame_number and start_frame != 0:
                            
                            # Calculating the angles
                            left_shoulder = keypoint[0][5][:2]
                            left_elbow = keypoint[0][7][:2]
                            left_wrist = keypoint[0][9][:2]
                            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                            left_hip = keypoint[0][11][:2]
                            left_knee = keypoint[0][13][:2]
                            left_ankle = keypoint[0][15][:2]
                            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                            
                            row = [keypoint[0][i][0] for i in range(1, 17)]  # x 좌표 저장
                            writer.writerow([frame_number, row, elbow_angle, knee_angle])

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