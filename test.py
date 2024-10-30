from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
results = model.track(source="./test/kim.mp4", show=True, save=True )