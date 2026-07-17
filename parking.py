"""Smart parking pipeline wrapper for YOLO-based slot detection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

from slot_detector_yolo import ParkingSlotYOLODetector


VIDEO_PATH = "data/videos/parking.mp4"
OUTPUT_PATH = "output/result.mp4"
SLOT_MODEL_PATH = "models/best.pt"

Detection = dict[str, Any]
Statistics = dict[str, int | float]


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata required to create an annotated output video."""

    width: int
    height: int
    fps: float
    frame_count: int


class SmartParkingPipeline:
    """Lightweight video pipeline around ParkingSlotYOLODetector."""

    WINDOW_NAME = "AI Smart Parking System"
    JPEG_EXTENSION = ".jpg"
    DEFAULT_FPS = 30.0

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the smart parking pipeline.

        Args:
            debug: Reserved for compatibility with previous integrations.

        Raises:
            RuntimeError: If the YOLO parking slot detector cannot initialize.
        """
        self.debug = debug
        self.slot_detector = self._create_slot_detector()
        self.latest_statistics: Statistics = {
            "total_slots": 0,
            "occupied_slots": 0,
            "available_slots": 0,
            "occupancy_percentage": 0.0,
            "fps": 0.0,
        }

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Process one frame using the YOLO parking slot detector.

        Args:
            frame: OpenCV BGR frame at native source resolution.

        Returns:
            Tuple containing the fully annotated frame and slot detections.

        Raises:
            RuntimeError: If frame processing fails.
        """
        start_time = time.perf_counter()

        try:
            annotated_frame, detections, statistics = (
                self.slot_detector.process_frame(frame)
            )
        except Exception as exc:
            raise RuntimeError(f"Frame processing failed: {exc}") from exc

        processing_fps = self._calculate_fps(start_time)
        statistics["fps"] = round(processing_fps, 1)
        self.latest_statistics = statistics

        return annotated_frame, detections

    def process_video(
        self,
        video_path: str | Path = VIDEO_PATH,
        output_path: str | Path = OUTPUT_PATH,
    ) -> None:
        """
        Process a video, display live annotations, and save annotated output.

        Args:
            video_path: Input video path.
            output_path: Output video path.

        Raises:
            RuntimeError: If the input video or output writer cannot be opened.
        """
        capture = self._open_video_capture(video_path)
        writer: cv2.VideoWriter | None = None

        try:
            metadata = self._read_video_metadata(capture)
            writer = self._create_video_writer(output_path, metadata)

            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    break

                annotated_frame, _detections = self.process_frame(frame)
                writer.write(annotated_frame)
                cv2.imshow(self.WINDOW_NAME, annotated_frame)

                if self._quit_requested():
                    break
        finally:
            self._release_resources(capture, writer)

    def generate_frames(
        self,
        video_path: str | Path = VIDEO_PATH,
    ) -> Generator[bytes, None, None]:
        """
        Generate multipart JPEG frames for Flask MJPEG streaming.

        The source video automatically loops when the end is reached.

        Args:
            video_path: Input video path.

        Yields:
            Multipart JPEG frame bytes.

        Raises:
            RuntimeError: If the input video cannot be opened.
        """
        capture = self._open_video_capture(video_path)

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    self._restart_video(capture)
                    continue

                annotated_frame, _detections = self.process_frame(frame)
                encoded_frame = self._encode_jpeg(annotated_frame)
                if encoded_frame is None:
                    continue

                yield self._build_mjpeg_frame(encoded_frame)
        finally:
            capture.release()

    @staticmethod
    def _create_slot_detector() -> ParkingSlotYOLODetector:
        """Create the YOLO parking slot detector."""
        try:
            return ParkingSlotYOLODetector(model_path=SLOT_MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"ParkingSlotYOLODetector initialization failed: {exc}"
            ) from exc

    @staticmethod
    def _open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
        """
        Open a video source.

        Args:
            video_path: Video file path.

        Returns:
            OpenCV VideoCapture instance.

        Raises:
            RuntimeError: If the video cannot be opened.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video: {video_path}")
        return capture

    @classmethod
    def _read_video_metadata(cls, capture: cv2.VideoCapture) -> VideoMetadata:
        """
        Read native video metadata from an opened capture.

        Args:
            capture: Opened OpenCV VideoCapture.

        Returns:
            Video metadata using the source resolution.

        Raises:
            RuntimeError: If the video dimensions are invalid.
        """
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid frame dimensions")

        if fps <= 0:
            fps = cls.DEFAULT_FPS

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    @staticmethod
    def _create_video_writer(
        output_path: str | Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        """
        Create a VideoWriter that preserves native video dimensions.

        Args:
            output_path: Destination video path.
            metadata: Source video metadata.

        Returns:
            Opened OpenCV VideoWriter.

        Raises:
            RuntimeError: If the writer cannot be opened.
        """
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

    @staticmethod
    def _calculate_fps(start_time: float) -> float:
        """
        Calculate instantaneous processing FPS.

        Args:
            start_time: perf_counter timestamp captured before processing.

        Returns:
            Frames processed per second.
        """
        elapsed = time.perf_counter() - start_time
        if elapsed <= 0:
            return 0.0
        return 1.0 / elapsed

    @staticmethod
    def _restart_video(capture: cv2.VideoCapture) -> None:
        """Restart a video capture from the first frame."""
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @classmethod
    def _encode_jpeg(cls, frame: np.ndarray) -> bytes | None:
        """
        Encode a frame as JPEG bytes.

        Args:
            frame: Annotated OpenCV frame.

        Returns:
            JPEG bytes, or None if encoding fails.
        """
        success, buffer = cv2.imencode(cls.JPEG_EXTENSION, frame)
        if not success:
            return None
        return buffer.tobytes()

    @staticmethod
    def _build_mjpeg_frame(jpeg_bytes: bytes) -> bytes:
        """
        Build one multipart MJPEG frame payload.

        Args:
            jpeg_bytes: Encoded JPEG frame.

        Returns:
            Multipart frame bytes suitable for Flask streaming.
        """
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )

    @staticmethod
    def _quit_requested() -> bool:
        """Return True when the user presses Q in the display window."""
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), ord("Q"))

    @staticmethod
    def _release_resources(
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter | None,
    ) -> None:
        """Release OpenCV resources used by video processing."""
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SmartParkingPipeline()
    pipeline.process_video(VIDEO_PATH, OUTPUT_PATH)
