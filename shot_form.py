import csv
import cv2
import numpy as np
from ultralytics import YOLO
from shot_detector import ShotDetector

class ShotForm():
    def __init__(self, player):
        # YOLOv8 Pose 모델 로드
        self.model = YOLO('yolo11n-pose.pt')

        self.cap = cv2.VideoCapture(f"./test/{player}.mp4")


        # 비디오 저장 설정 (프레임 크기 및 FPS 가져오기)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = f'./test_output/{player}.mp4'
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        

        self.frame_number = 0
        self.start_frame = 0
        self.end_frame = 0
        self.high_hand_y = 10000

        # CSV 파일 생성
        self.csv_filename = './output/curry.csv'
        with open(self.csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ['frame_number', 'keypoints', 'elbow_angle', 'knee_angle']
            csv_writer.writerow(header)
            
            # 허용 오차 설정
            height_tolerance = 0.005
            
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
            
            
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 프레임에서 포즈 추출
                results = self.model.track(source=frame, persist=True)
                
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
                            
                            # 결과 프레임 저장
                            self.out.write(frame)
                # 'q' 키를 누르면 종료
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
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

if __name__ == "__main__":
    ShotForm()