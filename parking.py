"""End-to-end video pipeline for the AI Smart Parking System."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Generator
from typing import Any

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_CONFIG_DIR = PROJECT_DIR / ".ultralytics"

VIDEO_PATH = "data/videos/parking.mp4"
OUTPUT_PATH = "output/parking_result.mp4"
PROCESSING_WIDTH = 1280
PROCESSING_HEIGHT = 720
DEBUG_OUTPUT_DIR = Path("output/debug")
MAX_SLOT_DETECTION_ATTEMPTS = 10


def _configure_ultralytics() -> None:
    """Use a project-local Ultralytics settings directory."""
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


_configure_ultralytics()

from detector import VehicleDetector
from occupancy_detector import OccupancyDetector
from slot_detector import ParkingSlotDetector


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata needed to create a matching output video."""

    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameProcessingResult:
    """Structured result for one processed frame."""

    annotated_frame: np.ndarray
    vehicle_detections: list[dict[str, Any]]
    parking_slots: list[tuple[int, int, int, int]]
    occupancy_data: list[dict[str, Any]]
    processing_fps: float


class SmartParkingPipeline:
    """Coordinate vehicle detection, slot detection, and occupancy analysis."""

    WINDOW_NAME = "AI Smart Parking System"
    OUTPUT_CODEC = "mp4v"
    VEHICLE_COLOR = (0, 255, 0)
    SLOT_COLOR = (0, 255, 0)
    FREE_OVERLAY_COLOR = (0, 255, 0)
    OCCUPIED_OVERLAY_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    TEXT_BACKGROUND_COLOR = (0, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        vehicle_detector: VehicleDetector | None = None,
        slot_detector: ParkingSlotDetector | None = None,
        occupancy_detector: OccupancyDetector | None = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the smart parking pipeline.

        Args:
            vehicle_detector: Optional preconfigured vehicle detector.
            slot_detector: Optional preconfigured parking slot detector.
            occupancy_detector: Optional preconfigured occupancy detector.
            debug: If True, save stage images to output/debug/.
        """
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.slot_detector = slot_detector or ParkingSlotDetector()
        self.occupancy_detector = occupancy_detector or OccupancyDetector()
        self.debug = debug
        self.cached_slots: list[tuple[int, int, int, int]] = []
        self.latest_statistics: dict[str, Any] = {}

        if self.debug:
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Process a single frame through the complete parking pipeline.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Tuple containing the annotated frame and occupancy records.

        Raises:
            RuntimeError: If a required detector stage fails.
        """
        if not self.cached_slots:
            resized_frame = self._resize_frame(frame)
            self.cached_slots = self._detect_parking_slots(resized_frame)
            result = self._process_frame_result(resized_frame)
            return result.annotated_frame, result.occupancy_data

        result = self._process_frame_result(frame, already_resized=False)
        return result.annotated_frame, result.occupancy_data

    def process_video(self, video_path: str | Path, output_path: str | Path) -> None:
        """
        Process a video, save annotated output, and display live results.

        Press Q in the display window to stop processing early.

        Args:
            video_path: Input video path.
            output_path: Output video path.

        Raises:
            FileNotFoundError: If the input video does not exist.
            OSError: If the video cannot be opened or the writer cannot be
                created.
        """
        input_path = Path(video_path)
        output_file = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise OSError(f"Unable to open input video: {input_path}")
        print("Video opened")

        metadata = self._read_video_metadata(capture)
        writer = self._create_video_writer(output_file, metadata)

        try:
            first_frame = self._initialize_cached_slots(capture)
            self._process_video_stream(capture, writer, first_frame)
        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()

    def generate_frames(
        self,
        video_path: str | Path,
    ) -> Generator[bytes, None, None]:
        """
        Generate processed JPEG frames for Flask MJPEG streaming.

        Args:
            video_path: Input video path.

        Yields:
            JPEG-encoded frame bytes formatted for MJPEG streaming.
        """

        input_path = Path(video_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input video not found: {input_path}"
            )

        capture = cv2.VideoCapture(str(input_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {input_path}"
            )

        frame_number = 0

        try:
            while True:

                success, frame = capture.read()

                # Restart video when it reaches the end
                if not success:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_number += 1

                try:
                    result = self._process_frame_result(
                        frame,
                        frame_number,
                        already_resized=False,
                    )

                    self.latest_statistics = (
                        self.occupancy_detector.get_statistics(
                            result.occupancy_data
                        )
                    )

                    success, buffer = cv2.imencode(
                        ".jpg",
                        result.annotated_frame,
                    )

                    if not success:
                        continue

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )

                except Exception as exc:
                    print(
                        f"[Frame {frame_number}] "
                        f"Processing error: {exc}"
                    )
                    continue

        finally:
            capture.release()

    def _initialize_cached_slots(self, capture: cv2.VideoCapture) -> np.ndarray:
        """
        Detect and cache static parking slots from the first valid frames.

        Args:
            capture: Open video capture positioned at the first frame.

        Returns:
            The resized frame where slot detection succeeded.

        Raises:
            RuntimeError: If no slots are detected after configured attempts.
        """
        for attempt in range(1, MAX_SLOT_DETECTION_ATTEMPTS + 1):
            success, frame = capture.read()
            if not success:
                break

            resized_frame = self._resize_frame(frame)

            try:
                slots = self._detect_parking_slots(resized_frame)
            except RuntimeError as exc:
                print(f"Slot detection attempt {attempt} failed: {exc}")
                continue

            if not slots:
                print(f"Slot detection attempt {attempt}: 0 slots")
                continue

            self.cached_slots = slots
            print(f"Parking slots detected: {len(self.cached_slots)}")

            if self.debug:
                self._save_initial_debug_images(resized_frame)

            return resized_frame

        raise RuntimeError(
            "Unable to detect parking slots after "
            f"{MAX_SLOT_DETECTION_ATTEMPTS} frame attempts."
        )

    def _process_frame_result(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        already_resized: bool = True,
    ) -> FrameProcessingResult:
        """Process one frame and return all stage outputs."""
        resized_frame = frame if already_resized else self._resize_frame(frame)
        start_time = time.perf_counter()

        vehicle_detections = self._detect_vehicles(resized_frame)
        occupancy_data = self._compute_occupancy(
            vehicle_detections,
            self.cached_slots,
        )
        self.latest_statistics = (
            self.occupancy_detector.get_statistics(
                occupancy_data
            )
        )

        processing_time = max(time.perf_counter() - start_time, 1e-9)
        processing_fps = 1.0 / processing_time
        annotated_frame = self._draw_pipeline_layers(
            frame=resized_frame,
            vehicle_detections=vehicle_detections,
            parking_slots=self.cached_slots,
            occupancy_data=occupancy_data,
            processing_fps=processing_fps,
        )

        if self.debug:
            self._save_frame_debug_images(
                resized_frame,
                vehicle_detections,
                occupancy_data,
                frame_number,
            )

        return FrameProcessingResult(
            annotated_frame=annotated_frame,
            vehicle_detections=vehicle_detections,
            parking_slots=self.cached_slots,
            occupancy_data=occupancy_data,
            processing_fps=processing_fps,
        )

    def _process_video_stream(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        first_frame: np.ndarray,
    ) -> None:
        """Read, process, display, and save frames until video end or Q."""
        frame_number = 1
        self._process_and_render_frame(first_frame, writer, frame_number)

        if self._quit_requested():
            return

        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            resized_frame = self._resize_frame(frame)

            if not self._process_and_render_frame(
                resized_frame,
                writer,
                frame_number,
            ):
                continue

            if self._quit_requested():
                break

    def _process_and_render_frame(
        self,
        frame: np.ndarray,
        writer: cv2.VideoWriter,
        frame_number: int,
    ) -> bool:
        """Process, write, display, and log one resized frame."""
        try:
            result = self._process_frame_result(
                frame,
                frame_number,
            )
            writer.write(result.annotated_frame)
            cv2.imshow(self.WINDOW_NAME, result.annotated_frame)
            self._log_progress(frame_number, result)
            return True
        except Exception as exc:
            print(f"Skipping frame {frame_number}: {exc}")
            return False

    def _detect_vehicles(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run vehicle detection with stage-specific error reporting."""
        try:
            detections = self.vehicle_detector.detect(frame)
            return detections if isinstance(detections, list) else []
        except Exception as exc:
            raise RuntimeError(f"YOLO vehicle detection failed: {exc}") from exc

    def _detect_parking_slots(
        self,
        frame: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Run parking slot detection with stage-specific error reporting."""
        try:
            slots = self.slot_detector.detect_slots(frame)
            return slots if isinstance(slots, list) else []
        except Exception as exc:
            raise RuntimeError(f"Parking slot detection failed: {exc}") from exc

    def _compute_occupancy(
        self,
        vehicle_detections: list[dict[str, Any]],
        parking_slots: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        """Run occupancy classification with stage-specific error reporting."""
        try:
            return self.occupancy_detector.check_occupancy(
                vehicle_detections,
                parking_slots,
            )
        except Exception as exc:
            raise RuntimeError(f"Occupancy calculation failed: {exc}") from exc

    def _draw_pipeline_layers(
        self,
        frame: np.ndarray,
        vehicle_detections: list[dict[str, Any]],
        parking_slots: list[tuple[int, int, int, int]],
        occupancy_data: list[dict[str, Any]],
        processing_fps: float,
    ) -> np.ndarray:
        """
        Draw layers in the required order.

        Order: original frame, vehicles, slots, occupancy, statistics, FPS.
        """
        annotated_frame = frame.copy()
        self._draw_vehicle_detections(annotated_frame, vehicle_detections)
        self._draw_parking_slots(annotated_frame, parking_slots)
        self._draw_occupancy_overlay(annotated_frame, occupancy_data)
        self._draw_statistics(annotated_frame, occupancy_data)
        self._draw_fps(annotated_frame, processing_fps)
        return annotated_frame

    def _draw_vehicle_detections(
        self,
        frame: np.ndarray,
        vehicle_detections: list[dict[str, Any]],
    ) -> None:
        """Draw green vehicle boxes and labels."""
        for detection in vehicle_detections:
            if not isinstance(detection, dict):
                continue

            bbox = self._safe_bbox(detection.get("bbox"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            label = self._vehicle_label(detection)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.VEHICLE_COLOR, 2)
            self._draw_text(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                self.VEHICLE_COLOR,
                0.55,
                2,
            )

    def _draw_parking_slots(
        self,
        frame: np.ndarray,
        parking_slots: list[tuple[int, int, int, int]],
    ) -> None:
        """Draw thin green parking slot outlines."""
        for slot in parking_slots:
            bbox = self._safe_bbox(slot)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.SLOT_COLOR, 1)

    def _draw_occupancy_overlay(
        self,
        frame: np.ndarray,
        occupancy_data: list[dict[str, Any]],
    ) -> None:
        """Draw red or green translucent occupancy overlays."""
        overlay = frame.copy()
        label_records: list[tuple[dict[str, Any], tuple[int, int, int]]] = []

        for record in occupancy_data:
            if not isinstance(record, dict):
                continue

            bbox = self._safe_bbox(record.get("bbox"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            occupied = bool(record.get("occupied", False))
            color = (
                self.OCCUPIED_OVERLAY_COLOR
                if occupied
                else self.FREE_OVERLAY_COLOR
            )

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
            label_records.append((record, color))

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

        for record, color in label_records:
            bbox = self._safe_bbox(record.get("bbox"))
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            self._draw_occupancy_label(frame, record, color)

    def _draw_occupancy_label(
        self,
        frame: np.ndarray,
        record: dict[str, Any],
        color: tuple[int, int, int],
    ) -> None:
        """Draw compact slot occupancy label."""
        bbox = self._safe_bbox(record.get("bbox"))
        if bbox is None:
            return

        x1, y1, _, _ = bbox
        slot_id = record.get("id", "?")
        occupied = bool(record.get("occupied", False))
        vehicle_class = record.get("vehicle_class") or "Vehicle"
        overlap = self._safe_float(record.get("overlap_score"))
        label = (
            f"Slot {slot_id} {str(vehicle_class).title()} {overlap * 100:.0f}%"
            if occupied
            else f"Slot {slot_id} Free"
        )
        self._draw_text(frame, label, (x1 + 4, max(20, y1 - 8)), color, 0.5, 2)

    def _draw_statistics(
        self,
        frame: np.ndarray,
        occupancy_data: list[dict[str, Any]],
    ) -> None:
        """Draw occupancy statistics at the top-left corner."""
        statistics = self.occupancy_detector.get_statistics(occupancy_data)
        lines = [
            f"Available: {statistics['available_slots']}",
            f"Occupied: {statistics['occupied_slots']}",
            f"Occupancy: {statistics['occupancy_percentage']:.1f}%",
        ]
        self._draw_text_block(frame, lines, (15, 30))

    def _draw_fps(self, frame: np.ndarray, processing_fps: float) -> None:
        """Draw processing FPS at the top-right corner."""
        text = f"FPS: {processing_fps:.1f}"
        text_size, _ = cv2.getTextSize(text, self.FONT, 0.75, 2)
        x = max(15, frame.shape[1] - text_size[0] - 15)
        self._draw_text(frame, text, (x, 30), self.TEXT_COLOR, 0.75, 2, True)

    def _save_initial_debug_images(self, first_frame: np.ndarray) -> None:
        """Save first-frame and cached-slot debug images."""
        first_frame_path = DEBUG_OUTPUT_DIR / "first_frame.jpg"
        cached_slots_path = DEBUG_OUTPUT_DIR / "cached_slots.jpg"

        cached_slots_frame = first_frame.copy()
        self._draw_parking_slots(cached_slots_frame, self.cached_slots)

        cv2.imwrite(str(first_frame_path), first_frame)
        cv2.imwrite(str(cached_slots_path), cached_slots_frame)

    def _save_frame_debug_images(
        self,
        frame: np.ndarray,
        vehicle_detections: list[dict[str, Any]],
        occupancy_data: list[dict[str, Any]],
        frame_number: int,
    ) -> None:
        """Save debug images for a specific frame."""

        occupancy_frame = frame.copy()

        self._draw_vehicle_detections(
            occupancy_frame,
            vehicle_detections,
        )

        self._draw_parking_slots(
            occupancy_frame,
            self.cached_slots,
        )

        self._draw_occupancy_overlay(
            occupancy_frame,
            occupancy_data,
        )

        self._draw_statistics(
            occupancy_frame,
            occupancy_data,
        )

        filename = (
            DEBUG_OUTPUT_DIR
            / f"occupancy_{frame_number:04d}.jpg"
        )

        cv2.imwrite(
            str(filename),
            occupancy_frame,
        )

    def _log_progress(
        self,
        frame_number: int,
        result: FrameProcessingResult,
    ) -> None:
        """Print processing progress every 30 frames."""

        if frame_number % 30 != 0:
            return

        statistics = self.occupancy_detector.get_statistics(
            result.occupancy_data
        )

        print(
            f"[Frame {frame_number}] "
            f"Vehicles: {len(result.vehicle_detections)} | "
            f"Occupied: {statistics['occupied_slots']} | "
            f"Available: {statistics['available_slots']} | "
            f"FPS: {result.processing_fps:.2f}"
        )

    @staticmethod
    def _read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
        """
        Read and validate video metadata from an open capture object.

        Args:
            capture: OpenCV video capture object.

        Returns:
            VideoMetadata containing fps and processing resolution.

        Raises:
            OSError: If frame dimensions cannot be read.
        """
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if source_width <= 0 or source_height <= 0:
            raise OSError("Unable to read input video dimensions.")

        if fps <= 0:
            fps = 30.0

        return VideoMetadata(
            fps=fps,
            width=PROCESSING_WIDTH,
            height=PROCESSING_HEIGHT,
        )

    def _create_video_writer(
        self,
        output_path: Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        """
        Create a VideoWriter using configured processing resolution.

        Args:
            output_path: Output file path.
            metadata: Output video metadata.

        Returns:
            OpenCV VideoWriter.

        Raises:
            OSError: If the output writer cannot be opened.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.OUTPUT_CODEC)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            metadata.fps,
            (metadata.width, metadata.height),
        )

        if not writer.isOpened():
            raise OSError(f"Unable to create output video: {output_path}")

        return writer

    @staticmethod
    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        """Resize a frame once to the configured processing resolution."""
        return cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))

    @staticmethod
    def _safe_bbox(value: Any) -> tuple[int, int, int, int] | None:
        """Normalize bbox-like values without raising."""
        try:
            if len(value) != 4:
                return None
            x1, y1, x2, y2 = (int(round(point)) for point in value)
        except (TypeError, ValueError):
            return None

        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    @staticmethod
    def _vehicle_label(detection: dict[str, Any]) -> str:
        """Build a readable vehicle label."""
        class_name = str(detection.get("class_name", "vehicle"))
        confidence = detection.get("confidence")

        if confidence is None:
            return class_name

        return f"{class_name} {SmartParkingPipeline._safe_float(confidence):.2f}"

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert a value to float, falling back to 0.0."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
    ) -> None:
        """Draw a readable text block with a black background."""
        x, y = origin
        line_height = 28
        padding = 8
        width = self._max_text_width(lines, font_scale=0.75, thickness=2)
        height = line_height * len(lines) + padding

        cv2.rectangle(
            frame,
            (x - padding, y - 22),
            (x + width + padding, y - 22 + height),
            self.TEXT_BACKGROUND_COLOR,
            thickness=-1,
        )

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + index * line_height),
                self.FONT,
                0.75,
                self.TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float,
        thickness: int,
        background: bool = False,
    ) -> None:
        """Draw text, optionally with a black background."""
        x, y = origin

        if background:
            text_size, baseline = cv2.getTextSize(
                text,
                self.FONT,
                font_scale,
                thickness,
            )
            cv2.rectangle(
                frame,
                (x - 8, y - text_size[1] - baseline - 8),
                (x + text_size[0] + 8, y + baseline + 8),
                self.TEXT_BACKGROUND_COLOR,
                thickness=-1,
            )

        cv2.putText(
            frame,
            text,
            origin,
            self.FONT,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _max_text_width(
        self,
        lines: list[str],
        font_scale: float,
        thickness: int,
    ) -> int:
        """Return maximum rendered text width for a set of lines."""
        if not lines:
            return 0

        return max(
            cv2.getTextSize(line, self.FONT, font_scale, thickness)[0][0]
            for line in lines
        )

    @staticmethod
    def _quit_requested() -> bool:
        """Return True when the user presses Q."""
        return (cv2.waitKey(1) & 0xFF) == ord("q")


def main() -> None:
    """Run the default smart parking video pipeline."""
    pipeline = SmartParkingPipeline()
    pipeline.process_video(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
