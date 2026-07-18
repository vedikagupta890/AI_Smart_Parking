from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

img = cv2.imread("data/images/2013-04-15_17_40_12_jpg.rf.23862bd082d4a0d25198927b625cd200.jpg")

if img is None:
    raise FileNotFoundError("Image not found.")

results = model(img)

print(results)

print(results[0].boxes.cls)
print(results[0].boxes.conf)