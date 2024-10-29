import csv
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 Pose 모델 로드
model = YOLO('yolov8n-pose.pt')

# 비디오 파일 열기
video_path = "./test/curry.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장 설정 (프레임 크기 및 FPS 가져오기)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = './test_output/output_video2.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

frame_number = 0
start_frame = 0
end_frame = 0
high_hand_y = 10000

# CSV 파일 생성
csv_filename = 'curry.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ['frame_number', 'keypoints', 'elbow_angle', 'knee_angle']
    csv_writer.writerow(header)
    
    # 허용 오차 설정
    height_tolerance = 0.01
    
    # 스켈레톤 연결 설정 (관절 인덱스 연결)
    skeleton_connections = [
        (5, 7), (7, 9),  # 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목
        (6, 8), (8, 10),  # 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목
        (5, 6),           # 왼쪽 어깨 - 오른쪽 어깨
        (11, 13), (13, 15), # 왼쪽 골반 - 왼쪽 무릎 - 왼쪽 발목
        (12, 14), (14, 16), # 오른쪽 골반 - 오른쪽 무릎 - 오른쪽 발목
        (11, 12),           # 왼쪽 골반 - 오른쪽 골반
        (5, 11), (6, 12)
    ]
    
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 번호 업데이트
        frame_number += 1
        
        # 프레임에서 포즈 추출
        results = model.track(source=frame, persist=True)
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            for i, box in enumerate(boxes):
                track_id = int(box.id) if box.id is not None else -1
                
                # track_id가 1인 객체만 처리
                if track_id == 2:
                    keypoint = keypoints[i].xyn.cpu().numpy()
                    key = keypoints[i].xy.cpu().numpy()
                    
                    
                    # 스켈레톤 그리기 (신뢰도가 높은 경우에만 그리기)
                    for start, end in skeleton_connections:
                        if keypoint[0][start][0] > 0 and keypoint[0][end][0] > 0:
                            start_point = (int(keypoint[0][start][0] * frame.shape[1]),
                                           int(keypoint[0][start][1] * frame.shape[0]))
                            end_point = (int(keypoint[0][end][0] * frame.shape[1]),
                                         int(keypoint[0][end][1] * frame.shape[0]))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                    
                    # 관절 높이 가져오기
                    left_shoulder_y = keypoint[0][5][1]  # 왼쪽 어깨의 y 좌표
                    left_elbow_y = keypoint[0][7][1]     # 왼쪽 팔꿈치의 y 좌표
                    right_shoulder_y = keypoint[0][6][1]  # 오른쪽 어깨의 y 좌표
                    right_elbow_y = keypoint[0][8][1]     # 오른쪽 팔꿈치의 y 좌표
                    right_hand_y = key[0][10][1]  # 오른쪽 손목의 y 좌표
                    left_hand_y = key[0][9][1]   # 왼쪽 손목의 y 좌표
                    
                    # 팔꿈치가 어깨 높이와 동일할 때 start_frame 설정
                    if (abs(left_elbow_y - left_shoulder_y) <= height_tolerance or
                        abs(right_elbow_y - right_shoulder_y) <= height_tolerance) and start_frame == 0:
                        start_frame = frame_number
                    
                    # start_frame 이후 손 위치가 가장 높을 때 업데이트
                    if start_frame != 0 :
                        if high_hand_y > min(right_hand_y, left_hand_y):
                            high_hand_y = min(right_hand_y, left_hand_y)
                            
                            # 스켈레톤 및 keypoint, 각도 데이터 계산 및 저장
                            cv2.imshow("High hand", frame)
                            
                            # # 왼쪽 팔꿈치 각도 계산
                            left_shoulder = keypoint[0][5][:2]
                            left_elbow = keypoint[0][7][:2]
                            left_wrist = keypoint[0][9][:2]
                            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                            
                            # 왼쪽 무릎 각도 계산
                            left_hip = keypoint[0][11][:2]
                            left_knee = keypoint[0][13][:2]
                            left_ankle = keypoint[0][15][:2]
                            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                            
                            # CSV에 키포인트 및 각도 기록
                            row = [frame_number, keypoint, elbow_angle, knee_angle]
                            csv_writer.writerow(row)
                            
                            # 결과 프레임 저장
                            out.write(frame)
        # cv2.imshow("skeleton", frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
