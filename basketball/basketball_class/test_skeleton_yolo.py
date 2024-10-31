import ast
import re
import cv2
import numpy as np
import pandas as pd

# 스켈레톤 연결 정의
skeleton_connections = [
    (5, 7), (7, 9),  # 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목
    (6, 8), (8, 10),  # 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목
    (5, 6),           # 왼쪽 어깨 - 오른쪽 어깨
    (11, 13), (13, 15), # 왼쪽 골반 - 왼쪽 무릎 - 왼쪽 발목
    (12, 14), (14, 16), # 오른쪽 골반 - 오른쪽 무릎 - 오른쪽 발목
    (11, 12),           # 왼쪽 골반 - 오른쪽 골반
    (5, 11), (6, 12)    # 왼쪽 어깨 - 왼쪽 골반, 오른쪽 어깨 - 오른쪽 골반
]

# 이미지에 스켈레톤 그리기 함수
def draw_skeleton(keypoints, img, skeleton_connections):
    height, width, _ = img.shape

    # 스켈레톤 그리기
    for start, end in skeleton_connections:
        if keypoints[start][0]> 0 and keypoints[end][0] > 0:
            
            # 정규화된 keypoints를 실제 이미지 크기로 변환
            x1, y1 = int(keypoints[start][0] * width), int(keypoints[start][1] * height)
            x2, y2 = int(keypoints[end][0] * width), int(keypoints[end][1] * height)

            # 선 그리기
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# CSV 파일 읽기
df = pd.read_csv('./basketball_player/james.csv')

# 문자열 표현을 실제 리스트로 변환하는 함수
def parse_keypoints(keypoints_str):
    # 문자열 정리: 앞뒤 공백 및 줄바꿈 제거
    
    # print(keypoints_str)  # 디버깅을 위해 출력
    keypoints_str = keypoints_str.strip().replace('\n', ' ')
    
    
    # 좌표 쌍 사이에 쉼표 삽입
    keypoints_str = re.sub(r'\[\s*([0-9.]+)\s+([0-9.]+)\s*\]', r'[\1,\2]', keypoints_str)
    
    # 리스트 요소들 사이에 쉼표 추가
    keypoints_str = re.sub(r'\]\s+\[', '], [', keypoints_str)
    

    # 불필요한 외부 괄호 정리
    keypoints_str = keypoints_str.replace('[[', '[').replace(']]', ']').replace('] [', '],[')

    
    
    return ast.literal_eval(keypoints_str)

# 키포인트 추출 및 리스트로 변환
df['keypoint'] = df['keypoint'].apply(parse_keypoints)

# 키포인트를 리스트 형태로 접근하기
keypoints_list = df.loc[df['Shooting'] == True, 'keypoint'].tolist()


# 이미지 크기 설정
img_height, img_width = 720, 1280

# 각 프레임의 키포인트로 스켈레톤 그리기
for keypoints_xy in keypoints_list:
    # 빈 이미지 생성
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # 각 프레임의 키포인트로 스켈레톤 그리기
    draw_skeleton(keypoints_xy, img, skeleton_connections)

    # 이미지 표시
    cv2.imshow("Skeleton", img)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # 'q'를 눌러 종료
        break

cv2.destroyAllWindows()