import cv2

video_path = "data/videos/parking.mp4"

cap = cv2.VideoCapture(video_path)

print("Opened:", cap.isOpened())

success, frame = cap.read()

print("Read frame:", success)

if success:
    print("Frame shape:", frame.shape)

cap.release()