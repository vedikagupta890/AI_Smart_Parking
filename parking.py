"""Smart parking video processing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

from detector import VehicleDetector
from occupancy_detector import OccupancyDetector
from slot_loader import load_slots, slots_to_bboxes


VIDEO_PATH = "data/videos/parking.mp4"
OUTPUT_PATH = "output/result.mp4"
SLOT_JSON_PATH = "data/slots/parking_slots.json"
PROCESSING_WIDTH = 1280
PROCESSING_HEIGHT = 720

BBox = tuple[int, int, int, int]
VehicleDetection = dict[str, Any]
OccupancyRecord = dict[str, Any]
Statistics = dict[str, int | float]


@dataclass(frozen=True)
class VideoMetadata:
    """Video source metadata used to configure output writers."""

    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class FrameProcessingResult:
    """Processed frame output and associated occupancy metadata."""

    annotated_frame: np.ndarray
    occupancy_data: list[OccupancyRecord]
    vehicle_detections: list[VehicleDetection]
    statistics: Statistics
    fps: float


class SmartParkingPipeline:
    """Run vehicle detection and occupancy estimation for fixed parking slots."""

    WINDOW_NAME = "AI Smart Parking System"
    JPEG_EXTENSION = ".jpg"
    DEFAULT_FPS = 30.0

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (0, 220, 255)

    def __init__(
        self,
        slot_json_path: str | Path = SLOT_JSON_PATH,
        debug: bool = False,
    ) -> None:
        """
        Initialize detectors and load fixed parking slots from JSON.

        Args:
            slot_json_path: Path to the parking slot configuration JSON.
            debug: Enables optional debug behavior for future integrations.

        Raises:
            RuntimeError: If detectors or parking slots cannot be initialized.
        """
        self.debug = debug
        self.slot_json_path = Path(slot_json_path)
        self.vehicle_detector = self._create_vehicle_detector()
        self.occupancy_detector = OccupancyDetector()
        self.cached_slots = self._load_cached_slots(self.slot_json_path)
        self.latest_statistics: Statistics = {
            "total_slots": len(self.cached_slots),
            "occupied_slots": 0,
            "available_slots": len(self.cached_slots),
            "occupancy_percentage": 0.0,
            "fps": 0.0,
        }

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[OccupancyRecord]]:
        """
        Process one video frame through the complete smart parking pipeline.

        Args:
            frame: OpenCV BGR frame.

        Returns:
            Tuple containing the annotated frame and occupancy records.
        """
        start_time = time.perf_counter()
        resized_frame = self._resize_frame(frame)
        vehicle_detections = self._detect_vehicles(resized_frame)
        occupancy_data = self._compute_occupancy(vehicle_detections)
        statistics = self.occupancy_detector.get_statistics(occupancy_data)
        processing_fps = self._calculate_fps(start_time)
        statistics["fps"] = round(processing_fps, 1)
        self.latest_statistics = statistics

        annotated_frame = self._draw_pipeline(
            resized_frame,
            vehicle_detections,
            occupancy_data,
            statistics,
            processing_fps,
        )
        return annotated_frame, occupancy_data

    def process_video(
        self,
        video_path: str | Path = VIDEO_PATH,
        output_path: str | Path = OUTPUT_PATH,
    ) -> None:
        """
        Process a video, save annotated output, and display live results.

        Args:
            video_path: Input video path.
            output_path: Annotated output video path.

        Raises:
            RuntimeError: If the video cannot be opened or writer fails.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video: {video_path}")

        writer: cv2.VideoWriter | None = None
        try:
            metadata = self._read_video_metadata(capture)
            writer = self._create_video_writer(output_path, metadata)

            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                annotated_frame, _occupancy_data = self.process_frame(frame)
                writer.write(annotated_frame)
                cv2.imshow(self.WINDOW_NAME, annotated_frame)

                if self._quit_requested():
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    def generate_frames(
        self,
        video_path: str | Path = VIDEO_PATH,
    ) -> Generator[bytes, None, None]:
        """
        Generate multipart JPEG frames for Flask MJPEG streaming.

        The video restarts automatically when the end is reached.

        Args:
            video_path: Input video path.

        Yields:
            Encoded multipart JPEG frame bytes.

        Raises:
            RuntimeError: If the video cannot be opened.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video: {video_path}")

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                annotated_frame, _occupancy_data = self.process_frame(frame)
                encode_success, buffer = cv2.imencode(
                    self.JPEG_EXTENSION,
                    annotated_frame,
                )
                if not encode_success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
        finally:
            capture.release()

    @staticmethod
    def _create_vehicle_detector() -> VehicleDetector:
        """Create the YOLO vehicle detector."""
        try:
            return VehicleDetector()
        except Exception as exc:
            raise RuntimeError(f"VehicleDetector initialization failed: {exc}") from exc

    @staticmethod
    def _load_cached_slots(slot_json_path: Path) -> list[BBox]:
        """Load fixed parking slot coordinates from JSON exactly once."""
        try:
            return slots_to_bboxes(load_slots(slot_json_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load parking slots from {slot_json_path}: {exc}"
            ) from exc

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the configured processing resolution."""
        return cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

    @staticmethod
    def _read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
        """Read metadata from an opened video capture."""
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = source_fps if source_fps > 0 else SmartParkingPipeline.DEFAULT_FPS
        return VideoMetadata(
            width=PROCESSING_WIDTH,
            height=PROCESSING_HEIGHT,
            fps=fps,
            frame_count=frame_count,
        )

    @staticmethod
    def _create_video_writer(
        output_path: str | Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        """Create a VideoWriter matching the processing resolution."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to create output video: {output_path}")
        return writer

    def _detect_vehicles(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run YOLO vehicle detection for one frame."""
        return self.vehicle_detector.detect(frame)

    def _compute_occupancy(
        self,
        vehicle_detections: list[VehicleDetection],
    ) -> list[OccupancyRecord]:
        """Compute occupancy for cached parking slots."""
        return self.occupancy_detector.check_occupancy(
            vehicle_detections,
            self.cached_slots,
        )

    def _draw_pipeline(
        self,
        frame: np.ndarray,
        vehicle_detections: list[VehicleDetection],
        occupancy_data: list[OccupancyRecord],
        statistics: Statistics,
        processing_fps: float,
    ) -> np.ndarray:
        """Draw all smart parking visualizations in the required order."""
        annotated_frame = frame.copy()
        self._draw_vehicle_boxes(annotated_frame, vehicle_detections)
        self._draw_slot_rectangles(annotated_frame, self.cached_slots)
        self._draw_occupancy_overlay(annotated_frame, occupancy_data)
        self._draw_statistics(annotated_frame, statistics)
        self._draw_processing_fps(annotated_frame, processing_fps)
        return annotated_frame

    def _draw_vehicle_boxes(
        self,
        frame: np.ndarray,
        vehicle_detections: list[VehicleDetection],
    ) -> None:
        """Draw vehicle bounding boxes and labels."""
        for detection in vehicle_detections:
            bbox = self._safe_bbox(detection.get("bbox"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.GREEN, 2)
            self._draw_text(
                frame,
                self._vehicle_label(detection),
                (x1, max(20, y1 - 8)),
                self.GREEN,
            )

    def _draw_slot_rectangles(
        self,
        frame: np.ndarray,
        parking_slots: list[BBox],
    ) -> None:
        """Draw cached parking slot outlines."""
        for slot_index, slot_bbox in enumerate(parking_slots, start=1):
            bbox = self._safe_bbox(slot_bbox)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.GREEN, 1)
            self._draw_text(
                frame,
                f"Slot {slot_index}",
                (x1 + 4, max(20, y1 - 6)),
                self.WHITE,
                scale=0.45,
                thickness=1,
            )

    def _draw_occupancy_overlay(
        self,
        frame: np.ndarray,
        occupancy_data: list[OccupancyRecord],
    ) -> None:
        """Draw transparent occupancy overlays for each parking slot."""
        overlay = frame.copy()

        for record in occupancy_data:
            bbox = self._safe_bbox(record.get("bbox"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            occupied = bool(record.get("occupied", False))
            color = self.RED if occupied else self.GREEN
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)

        for record in occupancy_data:
            bbox = self._safe_bbox(record.get("bbox"))
            if bbox is None:
                continue

            x1, y1, _x2, y2 = bbox
            slot_id = record.get("id", "?")
            status = "Occupied" if record.get("occupied", False) else "Free"
            color = self.RED if record.get("occupied", False) else self.GREEN
            self._draw_text(
                frame,
                f"{slot_id}: {status}",
                (x1 + 5, max(20, y2 - 8)),
                color,
                scale=0.48,
                thickness=1,
            )

    def _draw_statistics(
        self,
        frame: np.ndarray,
        statistics: Statistics,
    ) -> None:
        """Draw occupancy statistics in the top-left corner."""
        lines = [
            f"Available: {int(statistics.get('available_slots', 0))}",
            f"Occupied: {int(statistics.get('occupied_slots', 0))}",
            f"Occupancy: {self._safe_float(statistics.get('occupancy_percentage')):.1f}%",
        ]
        self._draw_text_block(frame, lines, (14, 30))

    def _draw_processing_fps(
        self,
        frame: np.ndarray,
        processing_fps: float,
    ) -> None:
        """Draw processing FPS in the top-right corner."""
        text = f"FPS: {processing_fps:.1f}"
        text_size, _baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        x_position = max(14, PROCESSING_WIDTH - text_size[0] - 16)
        self._draw_text(
            frame,
            text,
            (x_position, 30),
            self.YELLOW,
            scale=0.7,
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
        """Draw readable text on a frame."""
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            SmartParkingPipeline.BLACK,
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

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
    ) -> None:
        """Draw a compact block of text with a dark backing panel."""
        x_origin, y_origin = origin
        line_height = 28
        panel_width = 250
        panel_height = line_height * len(lines) + 16

        cv2.rectangle(
            frame,
            (x_origin - 8, y_origin - 24),
            (x_origin - 8 + panel_width, y_origin - 24 + panel_height),
            self.BLACK,
            thickness=-1,
        )

        for line_index, line in enumerate(lines):
            self._draw_text(
                frame,
                line,
                (x_origin, y_origin + line_index * line_height),
                self.WHITE,
            )

    @staticmethod
    def _safe_bbox(value: Any) -> BBox | None:
        """Normalize a bbox-like value into integer coordinates."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None

        try:
            x1, y1, x2, y2 = (int(coordinate) for coordinate in value)
        except (TypeError, ValueError):
            return None

        if x1 >= x2 or y1 >= y2:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert a value to float with a fallback."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _vehicle_label(self, detection: VehicleDetection) -> str:
        """Build a display label for a vehicle detection."""
        class_name = str(detection.get("class_name", "vehicle")).title()
        confidence = self._safe_float(detection.get("confidence"))
        return f"{class_name} {confidence:.2f}"

    @staticmethod
    def _quit_requested() -> bool:
        """Return True when the user requests exit from the display window."""
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), ord("Q"))

    @staticmethod
    def _calculate_fps(start_time: float) -> float:
        """Calculate instantaneous frame processing FPS."""
        elapsed = time.perf_counter() - start_time
        if elapsed <= 0:
            return 0.0
        return 1.0 / elapsed


if __name__ == "__main__":
    pipeline = SmartParkingPipeline()
    pipeline.process_video(VIDEO_PATH, OUTPUT_PATH)
