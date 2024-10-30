import ast
import re
import cv2
import numpy as np
import pandas as pd

# 스켈레톤 연결 정의
skeleton_connections = [
    (11, 13), (13, 15),                     # 왼쪽 팔
    (15, 17), (15, 21), (15, 19), (17, 19), # 왼쪽 손목과 손 관절  
    (12, 14), (14, 16),                     # 오른쪽 팔
    (16, 22), (16, 18), (16, 20), (18, 20), # 오른쪽 손목과 손 관절
    (11, 12), (11, 23), (23, 24), (12, 24), # 몸통
    (24, 26), (26, 28), (28, 32), (28, 30), (30, 32),   # 오른쪽 다리
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),   # 왼쪽 다리
]

# 이미지에 스켈레톤 그리기 함수
def draw_skeleton(keypoint, img, skeleton_connections):
    
    for pair in skeleton_connections:
        start_idx, end_idx = pair
        h, w, _ = img.shape
        if start_idx < len(keypoint) and end_idx < len(keypoint):
            start_point=(int(keypoint[start_idx][0] * w),
                            int(keypoint[start_idx][1] * h))
            end_point=(int(keypoint[end_idx][0] * w),
                            int(keypoint[end_idx][1] * h))
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)

# CSV 파일 읽기
df = pd.read_csv('shoot_angles.csv')

# 문자열 표현을 실제 리스트로 변환하는 함수
def parse_keypoint(keypoint_str):
    
    # print(keypoint_str)  # 디버깅을 위해 출력
    keypoint_str = keypoint_str.strip().replace('\'', ' ')
    
    return ast.literal_eval(keypoint_str)

# 키포인트 추출 및 리스트로 변환
df['keypoint'] = df['keypoint'].apply(parse_keypoint)

# 키포인트를 리스트 형태로 접근하기
keypoint_list = df['keypoint'].tolist()

# 이미지 크기 설정
img_height, img_width = 480, 640

# 각 프레임의 키포인트로 스켈레톤 그리기
for keypoint_xy in keypoint_list:
    # 빈 이미지 생성
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 각 프레임의 키포인트로 스켈레톤 그리기
    draw_skeleton(keypoint_xy, img, skeleton_connections)

    # 이미지 표시
    cv2.imshow("Skeleton", img)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # 'q'를 눌러 종료
        break

cv2.destroyAllWindows()