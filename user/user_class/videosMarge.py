import cv2
import numpy as np
from user.user_class.ball import ShotDetector
from user.user_class.shot_form import ShotForm
from user.dtw.main import DTW

class VideosMarge():
    def __init__(self, video_path, csv_filename, player):
        self.video_path=video_path
        self.csv_filename=csv_filename
        self.player=player
        
        
        
        self.filename = self.video_path.split(".")[0]
        self.result_player = self.filename.split("/")[1]

        ShotForm(self.video_path, self.csv_filename)
        ShotDetector(self.video_path, self.csv_filename)
        DTW(self.result_player, self.player)
        


        # 첫 번째 비디오 파일 열기
        self.cap1 = cv2.VideoCapture(f'user/user_output/{self.result_player}_ball.mp4')
        # 두 번째 비디오 파일 열기
        self.cap2 = cv2.VideoCapture(f'user/user_output/{self.result_player}.mp4')

        # 비디오 저장 설정 (프레임 크기 및 FPS 가져오기)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = int(self.cap1.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.output_path = f'user/user_marge/{self.result_player}.mp4'
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        while self.cap1.isOpened() and self.cap2.isOpened():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                break

            # 프레임 합성 (두 프레임을 가로로 합치기)
            combined_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            
            # 결과 프레임 저장
            self.out.write(combined_frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        self.out.release()
        cv2.destroyAllWindows()
        
if __name__=='__main__':
    VideosMarge()