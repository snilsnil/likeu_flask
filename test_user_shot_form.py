import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

class ShotForm():
    def __init__(self, player, csv_filename):
        # YOLOv8 Pose 모델 로드
        self.model = YOLO('yolo11n-pose.pt')

        self.player=player
        
        self.cap = cv2.VideoCapture(f"{self.player}")

        self.filename = self.player.split(".")[0]
        self.result_player = self.filename.split("/")[1]

        # 비디오 저장 설정 (프레임 크기 및 FPS 가져오기)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = f'user/user_output/{self.result_player}.mp4'
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.data_list = []

        self.frame_number = 0
        self.start_frame = 0
        self.end_frame = 0
        self.high_wrist_y = 10000
        self.previous_wrist_y=9999

        # 허용 오차 설정
        self.height_tolerance = 0.005
            
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
            
        self.findStartFrameAndEndFrame()    # 슛 시작점과 끝지점 찾기
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                # cv2로 열은 영상 프레임 초기화
        
        self.recordToCSV(skeleton_connections)                  # csv로 기록
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        df = pd.DataFrame(self.data_list)
        df.to_json(f'user/user_player/{csv_filename}', orient='records', indent=4)
        print(f"각도 데이터가 '{csv_filename}'로 저장되었습니다.")
        
        
        
        
    def startFrame(self, keypoint):
        
        left_elbow_y = keypoint[0][7][1]     # 왼쪽 팔꿈치의 y 좌표
        right_elbow_y = keypoint[0][8][1]     # 오른쪽 팔꿈치의 y 좌표
        right_wrist_y = keypoint[0][10][1]  # 오른쪽 손목의 y 좌표
        left_wrist_y = keypoint[0][9][1]   # 왼쪽 손목의 y 좌표
        left_ankle_x = keypoint[0][15][0]   # 왼쪽 발목 x 좌표
        right_ankle_x = keypoint[0][16][0]  # 오른쪽 발목
        
        # 팔꿈치가 어깨 높이와 동일할 때 start_frame 설정
        if self.start_frame == 0:
            if left_ankle_x > right_ankle_x and 0<abs(left_elbow_y - left_wrist_y) <= self.height_tolerance:
                self.start_frame = self.frame_number
            
            elif left_ankle_x < right_ankle_x and 0<abs(right_elbow_y - right_wrist_y) <= self.height_tolerance:
                self.start_frame = self.frame_number
    
    def endFrame(self, keypoint):
        
        right_wrist_y = keypoint[0][10][1]  # 오른쪽 손목의 y 좌표
        left_wrist_y = keypoint[0][9][1]   # 왼쪽 손목의 y 좌표
        left_ankle_x = keypoint[0][15][0]   # 왼쪽 발목 x 좌표
        right_ankle_x = keypoint[0][16][0]  # 오른쪽 발목
        
        if self.start_frame != 0 :
            if left_ankle_x > right_ankle_x and self.high_wrist_y > min(self.previous_wrist_y, left_wrist_y)>0:
                self.high_wrist_y = min(self.previous_wrist_y, left_wrist_y)
                self.previous_wrist_y=left_wrist_y
                self.end_frame = self.frame_number
            
            elif left_ankle_x < right_ankle_x and self.high_wrist_y > min(self.previous_wrist_y, right_wrist_y)>0:
                self.high_wrist_y = min(self.previous_wrist_y, right_wrist_y)
                self.previous_wrist_y=right_wrist_y
                self.end_frame = self.frame_number
    
    def leftElbowAngle(self,keypoint):
        left_shoulder = keypoint[0][5]
        left_elbow = keypoint[0][7]
        left_wrist = keypoint[0][9]
        return self.calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    def rightElbowAngle(self,keypoint):
        right_shoulder = keypoint[0][6]
        right_elbow = keypoint[0][8]
        right_wrist = keypoint[0][10]
        return self.calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    def leftKneeAngle(self,keypoint):
        left_hip = keypoint[0][11]
        left_knee = keypoint[0][13]
        left_ankle = keypoint[0][15]
        return self.calculate_angle(left_hip, left_knee, left_ankle)
    
    def rightKneeAngle(self,keypoint):
        right_hip = keypoint[0][12]
        right_knee = keypoint[0][14]
        right_ankle = keypoint[0][16]
        return self.calculate_angle(right_hip, right_knee, right_ankle)
    
    # 각도 계산 함수
    def calculate_angle(self, a, b, c):
        a = np.array(a)  # 첫 번째 점
        b = np.array(b)  # 중간 점
        c = np.array(c)  # 세 번째 점
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def recordShooting(self, frame, elbow, knee):
        if self.start_frame <= self.frame_number <= self.end_frame:
            cv2.imshow("frame", frame)
            self.data_list.append({
                'Frame': self.frame_number,
                'Elbow Angle': elbow,
                'Knee Angle': knee,
                'Shooting': True,
            })
        else:
            self.data_list.append({
                'Frame': self.frame_number,
                'Elbow Angle': elbow,
                'Knee Angle': knee,
                'Shooting': False,
            })
    
    def findStartFrameAndEndFrame(self):
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
                    if track_id == 1:
                        keypoint = keypoints[i].xyn.cpu().numpy()
                        
                        self.frame_number=self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        
                        self.startFrame(keypoint)
                        
                        self.endFrame(keypoint)
                        
                # 'q' 키를 누르면 종료
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
    def recordToCSV(self, skeleton_connections):
        results=self.model.predictor.trackers[0].reset()
        
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
                    
                    if track_id == 1:
                        keypoint = keypoints[i].xyn.cpu().numpy()
                        
                        # 스켈레톤 그리기 (신뢰도가 높은 경우에만 그리기)
                        for start, end in skeleton_connections:
                            if keypoint[0][start][0] > 0 and keypoint[0][end][0] > 0:
                                start_point = (int(keypoint[0][start][0] * frame.shape[1]),
                                                int(keypoint[0][start][1] * frame.shape[0]))
                                end_point = (int(keypoint[0][end][0] * frame.shape[1]),
                                                int(keypoint[0][end][1] * frame.shape[0]))
                                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                            
                        self.frame_number=self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            
                        left_elbow_angle=self.leftElbowAngle(keypoint)
                        right_elbow_angle=self.rightElbowAngle(keypoint)
                        left_knee_angle=self.leftKneeAngle(keypoint)
                        right_knee_angle=self.rightKneeAngle(keypoint)          
                        
                        left_ankle_x = keypoint[0][15][0]   # 왼쪽 발목 x 좌표
                        right_ankle_x = keypoint[0][16][0]  # 오른쪽 발목

                        if left_ankle_x > right_ankle_x:
                            self.recordShooting(frame, left_elbow_angle, left_knee_angle)
                        else:
                            self.recordShooting(frame, right_elbow_angle, right_knee_angle)
                        
                        # 결과 프레임 저장
                        self.out.write(frame)
                # 'q' 키를 누르면 종료
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
if __name__ == "__main__":
    player = 'uploads/20241101192135.MOV'
    csv = '20241101192135.json'
    ShotForm(player, csv)
