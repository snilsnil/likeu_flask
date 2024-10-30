import os
import cv2

video_path = "./uploads/1000002041.mp4"

# 파일 경로가 존재하는지 확인
if not os.path.exists(video_path):
    print(f"Error: 파일 경로가 존재하지 않습니다 - {video_path}")
else:
    print(f"파일이 존재합니다: {video_path}")

def show_frame_from(video_path, start_frame):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다.")
        return

    # 총 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 지정한 프레임이 범위를 벗어나는 경우 처리
    if start_frame >= total_frames:
        print(f"Error: 해당 비디오에는 {total_frames}개의 프레임만 있습니다.")
        return

    # 지정된 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 동영상의 끝에 도달했을 때 루프 종료

        # 프레임을 화면에 표시
        cv2.imshow("Video", frame)

        # 30ms 대기 후 다음 프레임으로 이동
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

    # 모든 창 닫기
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    start_frame = 9  # 비디오에서 시작할 프레임 번호
    show_frame_from(video_path, start_frame)
