import cv2
import csv
from ultralytics import YOLO

# YOLOv8 Pose 모델 로드
model = YOLO('yolov8n-pose.pt')

# 비디오 파일 열기
video_path = "./uploads/1000002041.mp4"
cap = cv2.VideoCapture(video_path)

# CSV 파일 생성
csv_filename = 'keypoints_data.csv'
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ['frame_number'] + [f'keypoint_{i}' for i in range(1, 17)]  # keypoint_0 제외
    csv_writer.writerow(header)

    # 허용 오차 설정
    height_tolerance = 0.01

    # 스켈레톤 연결 설정 (관절 인덱스 연결)
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

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 포즈 추출
        results = model(frame)

        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xyn.numpy()  # 키포인트 좌표 얻기
            frame_number += 1

            # 관절 높이 가져오기
            left_shoulder_y = keypoints[0][6][1]  # 왼쪽 어깨의 y 좌표
            left_elbow_y = keypoints[0][8][1]     # 왼쪽 팔꿈치의 y 좌표

            # 스켈레톤 그리기 (keypoint 0 제외, 신뢰도가 높은 경우에만 그리기)
            for start, end in skeleton_connections:
                if keypoints[0][start][0] > 0 and keypoints[0][end][0] > 0:  # x 좌표가 0보다 큰 경우
                    start_point = (int(keypoints[0][start][0] * frame.shape[1]),
                                   int(keypoints[0][start][1] * frame.shape[0]))
                    end_point = (int(keypoints[0][end][0] * frame.shape[1]),
                                 int(keypoints[0][end][1] * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # 선 그리기

            # 팔꿈치가 어깨 높이와 동일할 때
            if abs(left_elbow_y - left_shoulder_y) <= height_tolerance:
                print(f"Shooting started at frame: {frame_number}")

                # 해당 프레임을 화면에 표시
                cv2.imshow("Shooting Start Frame", frame)

                # 아무 키나 눌러야 계속 진행
                cv2.waitKey(0)  # 키 입력 대기
                break  # 비디오 정지

            # 키포인트 데이터를 CSV에 기록 (keypoint 0 제외)
            row = [frame_number] + [keypoints[0][i][0] for i in range(1, 17)]  # x 좌표 저장
            csv_writer.writerow(row)

        # 결과 프레임 표시
        cv2.imshow("Pose Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
