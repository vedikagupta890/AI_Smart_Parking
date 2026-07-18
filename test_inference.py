import cv2

from slot_detector_yolo import ParkingSlotYOLODetector

detector = ParkingSlotYOLODetector()

frame = cv2.imread(
    "data/images/2013-03-09_16_45_12_jpg.rf.e1b0585e629eb0c8edce9a03b543e097.jpg"
)

if frame is None:
    raise FileNotFoundError("Image not found.")

annotated_frame, detections, statistics = detector.process_frame(frame)

print(statistics)

cv2.imwrite("result.jpg", annotated_frame)

print("Saved result.jpg")

from collections import Counter

counter = Counter(d["class_id"] for d in detections)

print(counter)