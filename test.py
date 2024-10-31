from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
results = model.track(source="./test/james.mp4", show=True, save=True )