import cv2
import numpy as np
from shot_detector import ShotDetector
from shot_form import ShotForm

player="Booker"

ShotForm(player)
ShotDetector(player)

# 첫 번째 비디오 파일 열기
cap1 = cv2.VideoCapture(f'test_output/{player}_ball.mp4')
# 두 번째 비디오 파일 열기
cap2 = cv2.VideoCapture(f'test_output/{player}.mp4')

# 비디오 저장 설정 (프레임 크기 및 FPS 가져오기)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = './test_output/videoMerge.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # 프레임 합성 (두 프레임을 가로로 합치기)
    combined_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
    
    # 결과 프레임 저장
    out.write(combined_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
