"""
detector.py

Vehicle detection module for an AI Smart Parking System.

Tech Stack:
- Python 3.13
- OpenCV
- Ultralytics YOLO11
- NumPy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    Detect vehicles using a pretrained YOLO11 model.
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        """
        Initialize the vehicle detector.

        Args:
            model_name: Path or name of the YOLO model.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        try:
            self.model = YOLO(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_name}'."
            ) from exc

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect vehicles in a frame.

        Args:
            frame: OpenCV image (BGR).

        Returns:
            List of vehicle detections.

            Format:
            [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": (x1, y1, x2, y2)
                }
            ]
        """
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=0.25,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())

                if class_id not in self.VEHICLE_CLASSES:
                    continue

                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.VEHICLE_CLASSES[class_id],
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw vehicle detections on a frame.

        Args:
            frame: Original frame.
            detections: Vehicle detections.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            label = (
                f"{detection['class_name']} "
                f"{detection['confidence']:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            annotated,
            f"Vehicles: {len(detections)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: str,
    ) -> None:
        """
        Process an input video and save an annotated output video.

        Args:
            video_path: Input video path.
            output_path: Output video path.

        Raises:
            RuntimeError: If the input video cannot be opened.
        """
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            fps,
            (width, height),
        )

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                detections = self.detect(frame)

                annotated_frame = self.draw_detections(
                    frame,
                    detections,
                )

                writer.write(annotated_frame)

                cv2.imshow(
                    "Vehicle Detection",
                    annotated_frame,
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = VehicleDetector()

    detector.process_video(
        "data/videos/parking.mp4",
        "output/result.mp4",
    )