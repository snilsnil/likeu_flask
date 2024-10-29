import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# MediaPipe 포즈 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 파일 열기
cap = cv2.VideoCapture('./test/curry.mp4')

# 농구공 궤적을 저장할 리스트
ball_trajectory = []
shoot_start_points = []  # 슛 시작 지점을 저장할 리스트
data_list = []  # 각도 데이터를 저장할 리스트

if not cap.isOpened():
    print("Error: 비디오 파일을 열 수 없습니다.")
    exit()

# 포즈 좌표를 위한 이전 위치 저장
previous_landmarks = []
shooting = False  # 슛 상태
shoot_start_frame = None  # 슛 시작 프레임
max_hand_y=100000
frame_number=None

# 스켈레톤 연결 인덱스 정의 (머리와 얼굴 키포인트 제외)
skeleton_pairs = [
    (11, 13), (13, 15),                     # 왼쪽 팔
    (15, 17), (15, 21), (15, 19), (17, 19), # 왼쪽 손목과 손 관절  
    (12, 14), (14, 16),                     # 오른쪽 팔
    (16, 22), (16, 18), (16, 20), (18, 20), # 오른쪽 손목과 손 관절
    (11, 12), (11, 23), (23, 24), (12, 24), # 몸통
    (24, 26), (26, 28), (28, 32), (28, 30), (30, 32),   # 오른쪽 다리
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),   # 왼쪽 다리
]

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중간 점
    c = np.array(c)  # 세 번째 점
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추정
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_pose = pose.process(frame_rgb)

    # 포즈 시각화 (머리와 얼굴 키포인트 제외)
    if result_pose.pose_landmarks:
        # 이전 랜드마크와 현재 랜드마크의 위치를 비교하여 부드럽게 그리기
        current_landmarks = []
        for i, landmark in enumerate(result_pose.pose_landmarks.landmark):
            h, w, _ = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            current_landmarks.append((cx, cy))


            # 관절 위치 표시 (머리와 얼굴 키포인트 제외)
            if i not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 머리(0, 1)와 얼굴(2, 4, 5, 6) 키포인트 제외
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # 관절 위치 표시

        # 스켈레톤
        for pair in skeleton_pairs:
            start_idx, end_idx = pair
            if start_idx < len(current_landmarks) and end_idx < len(current_landmarks):
                cv2.line(frame, current_landmarks[start_idx], current_landmarks[end_idx], (255, 0, 0), 2)
        # 팔꿈치 각도 계산
        left_shoulder = current_landmarks[11]
        left_elbow = current_landmarks[13]
        left_hand = current_landmarks[15]

        right_shoulder = current_landmarks[12]
        right_elbow = current_landmarks[14]
        right_hand = current_landmarks[16]

        # 왼팔과 오른팔의 각도 계산
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_hand)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_hand)

        # 무릎 각도 계산 (왼쪽 무릎과 오른쪽 무릎)
        left_knee = current_landmarks[23]
        right_knee = current_landmarks[24]
        body_mid = current_landmarks[11]  # 몸통 위치

        left_knee_angle = calculate_angle(body_mid, left_knee, (left_knee[0], left_knee[1] + 100))
        right_knee_angle = calculate_angle(body_mid, right_knee, (right_knee[0], right_knee[1] + 100))

        # 슛 시작 지점 판별 (팔꿈치 각도가 90도 이상인 경우)
        if left_elbow_angle < 90 or right_elbow_angle < 90:
            if not shooting and shoot_start_frame is None:
                shooting = True
                shoot_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # 슛 종료 지점 판별 (손의 높이를 기준으로)
        if shooting and shoot_start_frame is not None :
            if max_hand_y > min(left_hand[1], right_hand[1]):
                max_hand_y = min(max_hand_y, min(left_hand[1], right_hand[1]))
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cv2.imshow("skeleton", frame)
                
                # 각도 데이터를 리스트에 저장
                data_list.append({
                    'Frame': frame_number,
                    'Left Elbow Angle': left_elbow_angle,
                    'Right Elbow Angle': right_elbow_angle,
                    'Left Knee Angle': left_knee_angle,
                    'Right Knee Angle': right_knee_angle,
                })
                
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# CSV 파일로 저장
df = pd.DataFrame(data_list)
df.to_csv('shoot_angles.csv', index=False)
print("각도 데이터가 'shoot_angles.csv'로 저장되었습니다.")

cv2.destroyAllWindows()
