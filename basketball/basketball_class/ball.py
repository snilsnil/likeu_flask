# Import libraries
from ultralytics import YOLO
import pandas as pd
import cv2
import cvzone
import math
import numpy as np
import imageio
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector():
    def __init__(self, player):
        # 모델, 비디오, 및 클래스 초기화
        self.player = player  # player를 인스턴스 변수로 저장
        self.model = YOLO("models/shot_detector_modle.pt")
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.cap = cv2.VideoCapture(f"test/{player}.mp4")
        
        # 비디오 설정
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = f'basketball/basketball_output/{player}_ball.mp4'
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        # 위치 변수 초기화
        self.ball_pos = []
        self.hoop_pos = []
        self.gif_images = []  # GIF 프레임 저장을 위한 리스트
        self.data_list = []   # csv 파일 저장을 위한 리스트

        # 프레임 및 궤적 이미지 초기화
        self.frame_count = 0
        self.frame = None
        self.trajectory_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 점수 및 시각화 초기화
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # 비디오 분석 실행
        self.run()
        
        df = pd.DataFrame(self.data_list)
        df.to_json(f'basketball/basketball_ball/{self.player}.json', orient='records', indent=4)
        print(f"공 데이터가 'basketball/basketball_ball/{self.player}.json'로 저장되었습니다.")

    def run(self):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            if self.frame_count % 1 == 0:  # 매 프레임마다 처리
                results = self.model(self.frame, stream=True)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        current_class = self.class_names[cls]
                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # 공 위치 추가
                        if (conf > .4 or (in_hoop_region(center, self.hoop_pos) and conf > 0.2)) and current_class == "Basketball": #이거 조정해야댐
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                            cv2.circle(self.trajectory_image, center, 2, (0, 0, 255), -1)  # 공 위치 표시

                        # 후프 위치 추가
                        if conf > .2 and current_class == "Basketball Hoop":
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))
                            cv2.circle(self.trajectory_image, center, 2, (128, 128, 0), -1)
                            
            self.frame_count = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            

            # GIF 프레임 추가
            gif_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for pos in self.ball_pos:
                cv2.circle(gif_frame, pos[0], 5, (0, 0, 255), -1)  # 공 위치를 빨간색으로 표시
            self.gif_images.append(gif_frame)

            self.out.write(self.frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        # 비디오와 창 닫기
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        # 궤적 이미지 및 GIF 저장
        self.save_trajectory_image()

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
    
        # 임계값 설정 (너무 큰 이동을 무시)
        distance_threshold = 70  # 픽셀 거리 이거 해상도에 따라 다른데 폰이면 100정도 하면됨
        for i in range(1, len(self.ball_pos)):
            current_pos = self.ball_pos[i][0]
            previous_pos = self.ball_pos[i-1][0]
        
            # 거리가 threshold를 넘는 경우 무시
            if np.linalg.norm(np.array(current_pos) - np.array(previous_pos)) < distance_threshold:
                cv2.circle(self.frame, current_pos, 2, (0, 0, 255), 2)
                
            
            ball_xyn = self.xy_to_xyn(current_pos)
        
            self.data_list.append({
                    'Frame': self.frame_count,
                    'keypoint' : ball_xyn,
                })

        # 후프 위치 그리기
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)
            
    def xy_to_xyn(self, keypoint):
        x, y = keypoint
        xn = x / self.frame.shape[1]
        yn = y / self.frame.shape[0]
        return [xn, yn]
        

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames
                        # 슛 성공 시 공을 초록색으로 표시
                        for pos, _, _, _, _ in self.ball_pos:
                            cv2.circle(self.frame, pos, 2, (0, 255, 0), -1)
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

    def display_score(self):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

    def save_trajectory_image(self):
        # 궤적 이미지 저장
        trajectory_path = f"basketball/basketball_output/{self.player}_trajectory.png"
        cv2.imwrite(trajectory_path, self.trajectory_image)
        print(f"공의 궤적 이미지가 {trajectory_path}에 저장되었습니다.")
        
        # GIF 저장
        gif_path = f"basketball/basketball_output/{self.player}_trajectory.gif"
        imageio.mimsave(gif_path, self.gif_images, duration=0.1)  # duration은 프레임 간 시간, 초 단위
        print(f"공의 궤적 GIF가 {gif_path}에 저장되었습니다.")

if __name__ == "__main__":
    ShotDetector(player="Booker")
