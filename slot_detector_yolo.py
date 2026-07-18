"""YOLO-based parking slot detection for the AI Smart Parking System."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class VideoMetadata:
    """Video metadata required to configure an output writer."""

    width: int
    height: int
    fps: float


class ParkingSlotYOLODetector:
    """Detect free and occupied parking slots using a custom YOLO11 model."""

    CLASS_NAMES = {
        0: "free",
        1: "occupied",
    }

    FREE_COLOR = (0, 255, 0)
    OCCUPIED_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    PANEL_COLOR = (0, 0, 0)
    DEFAULT_FPS = 30.0
    WINDOW_NAME = "YOLO Parking Slot Detection"

    def __init__(
        self,
        model_path: str = "models/best.onnx",
        confidence_threshold: float = 0.25,
    ) -> None:
        """
        Initialize the YOLO parking slot detector.

        Args:
            model_path: Path to the custom-trained YOLO11 parking slot model.
            confidence_threshold: Minimum confidence required for detections.

        Raises:
            ValueError: If confidence_threshold is outside [0.0, 1.0].
            RuntimeError: If the YOLO model cannot be loaded.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = self._validate_confidence_threshold(
            confidence_threshold
        )
        self.model = self._load_model(self.model_path)
        import numpy as np

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        self.model.predict(
            source=dummy,
            device="cpu",
            verbose=False,
            imgsz=640,
        )

    def detect_slots(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect parking slots in a single frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            List of slot detection dictionaries.

        Raises:
            ValueError: If the input frame is invalid.
            RuntimeError: If YOLO inference fails.
        """
        self._validate_frame(frame)

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=640,
                device="cpu",
                verbose=False,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                detection = self._box_to_detection(box)
                if detection is not None:
                    detections.append(detection)

        return self._assign_slot_ids(detections)

    def draw_slots(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Draw detected parking slots on a frame.

        Args:
            frame: OpenCV BGR image.
            detections: Slot detections returned by detect_slots().

        Returns:
            Annotated frame.

        Raises:
            ValueError: If the input frame is invalid.
        """
        self._validate_frame(frame)
        annotated_frame = frame.copy()

        for detection in detections:
            if not self._is_valid_detection(detection):
                continue

            bbox = detection["bbox"]
            color = self._slot_color(bool(detection["occupied"]))
            cv2.rectangle(
                annotated_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2,
            )
            self._draw_detection_label(annotated_frame, detection, color)

        return annotated_frame

    def get_statistics(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, int | float]:
        """
        Calculate parking slot occupancy statistics.

        Args:
            detections: Slot detections returned by detect_slots().

        Returns:
            Dictionary containing total, occupied, available, and percentage.
        """
        valid_detections = [
            detection
            for detection in detections
            if self._is_valid_detection(detection)
        ]
        total_slots = len(valid_detections)
        occupied_slots = sum(
            1 for detection in valid_detections if detection["occupied"]
        )
        available_slots = total_slots - occupied_slots
        occupancy_percentage = (
            (occupied_slots / total_slots) * 100.0 if total_slots else 0.0
        )

        return {
            "total_slots": total_slots,
            "occupied_slots": occupied_slots,
            "available_slots": available_slots,
            "occupancy_percentage": round(occupancy_percentage, 1),
        }

    def draw_statistics(
        self,
        frame: np.ndarray,
        statistics: dict[str, int | float],
    ) -> np.ndarray:
        """
        Draw occupancy statistics in a semi-transparent panel.

        Args:
            frame: OpenCV BGR image.
            statistics: Statistics returned by get_statistics().

        Returns:
            Annotated frame.

        Raises:
            ValueError: If the input frame is invalid.
        """
        self._validate_frame(frame)
        annotated_frame = frame.copy()
        lines = [
            f"Available: {int(statistics.get('available_slots', 0))}",
            f"Occupied: {int(statistics.get('occupied_slots', 0))}",
            "Occupancy: "
            f"{float(statistics.get('occupancy_percentage', 0.0)):.1f}%",
        ]
        self._draw_summary_panel(annotated_frame, lines)
        return annotated_frame

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int | float]]:
        """
        Detect, annotate, and summarize parking slots in one frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            Tuple of annotated frame, detections, and statistics.
        """
        detections = self.detect_slots(frame)
        statistics = self.get_statistics(detections)
        annotated_frame = self.draw_slots(frame, detections)
        annotated_frame = self.draw_statistics(annotated_frame, statistics)
        return annotated_frame, detections, statistics
    
    def process_image(
        self,
        image_path: str,
        output_path: str,
    ) -> None:
        """
        Detect parking slots in a single image and save the result.

        Args:
            image_path: Input image path.
            output_path: Output image path.
        """

        frame = cv2.imread(image_path)

        if frame is None:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        annotated_frame, detections, statistics = self.process_frame(frame)

        print("\nDetections:")
        for detection in detections:
            print(detection)

        print("\nStatistics:")
        print(statistics)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_file), annotated_frame)

        cv2.imshow("Parking Slot Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"\nOutput saved to: {output_file}")

        
    def process_video(
        self,
        input_video: str | Path,
        output_video: str | Path,
    ) -> None:
        """
        Run YOLO slot detection on a video and save annotated output.

        Args:
            input_video: Source video path.
            output_video: Destination video path.

        Raises:
            RuntimeError: If the video cannot be opened or written.
        """
        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open input video: {input_video}")

        writer: cv2.VideoWriter | None = None
        try:
            metadata = self._read_video_metadata(capture)
            writer = self._create_video_writer(output_video, metadata)

            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                annotated_frame, _detections, _statistics = self.process_frame(
                    frame
                )
                writer.write(annotated_frame)
                cv2.imshow(self.WINDOW_NAME, annotated_frame)

                if self._quit_requested():
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    @staticmethod
    def _validate_confidence_threshold(threshold: float) -> float:
        """Validate and normalize a confidence threshold."""
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_threshold must be numeric") from exc

        if not 0.0 <= threshold_value <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        return threshold_value

    @staticmethod
    def _load_model(model_path: Path) -> YOLO:
        try:
            model = YOLO(
                    str(model_path),
                    task="detect",
                )

            # Only PyTorch models support .to()
            if model_path.suffix == ".pt":
                model.to("cpu")

            return model

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO parking slot model '{model_path}'."
            ) from exc

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate an OpenCV frame."""
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("frame must not be empty")

        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a grayscale or BGR image")

    def _box_to_detection(self, box: Any) -> dict[str, Any] | None:
        """Convert one YOLO box into a normalized detection dictionary."""
        class_id = self._extract_class_id(box)
        if class_id not in self.CLASS_NAMES:
            return None

        confidence = self._extract_confidence(box)
        if confidence < self.confidence_threshold:
            return None

        bbox = self._extract_bbox(box)
        if bbox is None:
            return None

        class_name = self.CLASS_NAMES[class_id]
        return {
            "slot_id": 0,
            "bbox": bbox,
            "occupied": class_name == "occupied",
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name,
        }

    @staticmethod
    def _extract_class_id(box: Any) -> int:
        """Extract class ID from a YOLO box."""
        return int(box.cls.item())

    @staticmethod
    def _extract_confidence(box: Any) -> float:
        """Extract confidence from a YOLO box."""
        return float(box.conf.item())

    @staticmethod
    def _extract_bbox(box: Any) -> tuple[int, int, int, int] | None:
        """Extract and validate xyxy coordinates from a YOLO box."""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        except (TypeError, ValueError, IndexError, AttributeError):
            return None

        if x1 >= x2 or y1 >= y2:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _assign_slot_ids(
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assign stable sequential IDs ordered top-to-bottom, left-to-right."""
        sorted_detections = sorted(
            detections,
            key=lambda detection: (
                detection["bbox"][1],
                detection["bbox"][0],
            ),
        )

        for slot_id, detection in enumerate(sorted_detections, start=1):
            detection["slot_id"] = slot_id

        return sorted_detections

    def _is_valid_detection(self, detection: dict[str, Any]) -> bool:
        """Return True if a detection has the expected shape and values."""
        if not isinstance(detection, dict):
            return False

        bbox = detection.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            return False

        if not all(isinstance(value, int) for value in bbox):
            return False

        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return False

        class_id = detection.get("class_id")
        if class_id not in self.CLASS_NAMES:
            return False

        confidence = detection.get("confidence")
        if not isinstance(confidence, (int, float)):
            return False

        return float(confidence) >= self.confidence_threshold

    def _format_detection_label(
        self,
        detection: dict[str, Any],
    ) -> list[str]:
        """Format multi-line label text for a slot detection."""
        status = str(detection["class_name"]).title()
        confidence_percent = int(round(float(detection["confidence"]) * 100))
        return [
            f"Slot {int(detection['slot_id'])}",
            status,
            f"{confidence_percent}%",
        ]

    def _slot_color(self, occupied: bool) -> tuple[int, int, int]:
        """Return the drawing color for a slot state."""
        return self.OCCUPIED_COLOR if occupied else self.FREE_COLOR

    def _draw_detection_label(
        self,
        frame: np.ndarray,
        detection: dict[str, Any],
        color: tuple[int, int, int],
    ) -> None:
        """Draw a detection label near a bounding box."""
        x1, y1, _x2, _y2 = detection["bbox"]
        label_lines = self._format_detection_label(detection)
        self._draw_label_block(
            frame=frame,
            lines=label_lines,
            origin=(x1, max(20, y1 - 54)),
            accent_color=color,
        )

    def _draw_label_block(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
        accent_color: tuple[int, int, int],
    ) -> None:
        """Draw a compact multi-line text label."""
        x_origin, y_origin = origin
        line_height = 18

        for index, line in enumerate(lines):
            self._draw_text(
                frame=frame,
                text=line,
                origin=(x_origin, y_origin + index * line_height),
                color=accent_color if index == 0 else self.TEXT_COLOR,
                scale=0.5,
                thickness=1,
            )

    def _draw_summary_panel(
        self,
        frame: np.ndarray,
        lines: list[str],
    ) -> None:
        """Draw the upper-left statistics panel."""
        overlay = frame.copy()
        x1, y1 = 12, 12
        line_height = 28
        panel_width = 260
        panel_height = line_height * len(lines) + 18

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x1 + panel_width, y1 + panel_height),
            self.PANEL_COLOR,
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)

        for index, line in enumerate(lines):
            self._draw_text(
                frame=frame,
                text=line,
                origin=(x1 + 12, y1 + 28 + index * line_height),
                color=self.TEXT_COLOR,
                scale=0.65,
                thickness=2,
            )

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
        """Draw anti-aliased text with a dark outline for readability."""
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def _read_video_metadata(cls, capture: cv2.VideoCapture) -> VideoMetadata:
        """Read video dimensions and FPS from an opened capture."""
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid frame dimensions")

        if fps <= 0:
            fps = cls.DEFAULT_FPS

        return VideoMetadata(width=width, height=height, fps=fps)

    @staticmethod
    def _create_video_writer(
        output_video: str | Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        """Create a VideoWriter for annotated output."""
        output_path = Path(output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to create output video: {output_video}")

        return writer

    @staticmethod
    def _quit_requested() -> bool:
        """Return True when the user presses Q."""
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), ord("Q"))


if __name__ == "__main__":
    detector = ParkingSlotYOLODetector(
        model_path="models/best.onnx"
    )

    # detector.process_image(
    #     "data/images/2013-03-09_16_45_12_jpg.rf.e1b0585e629eb0c8edce9a03b543e097.jpg",
    #     "output/test_result.jpg",
    # )

    detector.process_video(
    "data/videos/parking.mp4",
    "output/parking_result.mp4"
    )