from ultralytics import YOLO

model = YOLO("models/best.onnx")

results = model.predict(
    source="data/videos/parking.mp4",
    save=True,
    conf=0.25,
)

print("Inference completed successfully!")