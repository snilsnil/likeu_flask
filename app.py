from flask import Flask, jsonify, request, Response, render_template, send_file
from ultralytics import YOLO
import numpy as np
import cv2
import os

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

# 스켈레톤 연결 설정
skeleton_connections = [
    (5, 6),  # 오른쪽 어깨 - 왼쪽 어깨
    (5, 7),  # 오른쪽 어깨 - 오른쪽 팔꿈치
    (6, 8),  # 왼쪽 어깨 - 왼쪽 팔꿈치
    (7, 9),  # 오른쪽 팔꿈치 - 오른쪽 손목
    (8, 10), # 왼쪽 팔꿈치 - 왼쪽 손목
    (5, 11), # 오른쪽 어깨 - 오른쪽 골반
    (6, 12), # 왼쪽 어깨 - 왼쪽 골반
    (11, 13),# 오른쪽 골반 - 오른쪽 무릎
    (12, 14),# 왼쪽 골반 - 왼쪽 무릎
    (11, 15),# 오른쪽 골반 - 오른쪽 발
    (12, 16) # 왼쪽 골반 - 왼쪽 발
]

# 비디오 처리 함수
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임에서 포즈 추출
        results = model(frame)
        keypoints = results[0].keypoints.xyn.numpy()
        boxes = results[0].boxes.xyxy.numpy()

        max_area = 0
        selected_person = None

        for person, box in zip(keypoints, boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                selected_person = person

        if selected_person is not None:
            # 스켈레톤 그리기
            for start, end in skeleton_connections:
                start_point = (int(selected_person[start][0] * frame.shape[1]), int(selected_person[start][1] * frame.shape[0]))
                end_point = (int(selected_person[end][0] * frame.shape[1]), int(selected_person[end][1] * frame.shape[0]))
                if selected_person[start][0] > 0 and selected_person[start][1] > 0 and selected_person[end][0] > 0 and selected_person[end][1] > 0:
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # 선 그리기

            # 프레임 표시
            cv2.imshow("Skeleton Detection", frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

# 비디오 업로드 및 처리 라우트
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected video", 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # 비디오 처리
    process_video(video_path)

    return "Video processed successfully", 200

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True, port=3000, host='0.0.0.0')
