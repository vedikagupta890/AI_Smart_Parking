# import cv2

# print(cv2.__version__)

# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")

# print("YOLO loaded successfully!")

# import cv2

# print(cv2.__version__)

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# print(fourcc)


import cv2

print(cv2.__version__)
print(cv2.__file__)
print(hasattr(cv2, "VideoWriter_fourcc"))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
print(fourcc)