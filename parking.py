"""Reusable video pipeline for the AI Smart Parking System."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np

from slot_detector_yolo import ParkingSlotYOLODetector


logger = logging.getLogger(__name__)

VIDEO_PATH = "data/videos/parking.mp4"
OUTPUT_PATH = "output/result.mp4"
SLOT_MODEL_PATH = "models/best.pt"

Detection = dict[str, Any]
Statistics = dict[str, int | float]


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata required to configure an output video writer."""

    width: int
    height: int
    fps: float
    frame_count: int


class SmartParkingPipeline:
    """Video processing wrapper around ParkingSlotYOLODetector."""

    WINDOW_NAME = "AI Smart Parking System"
    JPEG_EXTENSION = ".jpg"
    DEFAULT_FPS = 30.0

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the smart parking video pipeline.

        Args:
            debug: If True, enables local OpenCV preview windows.

        Raises:
            RuntimeError: If the slot detector cannot be initialized.
        """
        self.debug = debug
        self.slot_detector = self._initialize_detector()
        self.latest_statistics: Statistics = {
            "total_slots": 0,
            "occupied_slots": 0,
            "available_slots": 0,
            "occupancy_percentage": 0.0,
            "fps": 0.0,
        }
        logger.info("Pipeline initialized")

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Process one frame with the YOLO parking slot detector.

        Args:
            frame: OpenCV BGR frame.

        Returns:
            Annotated frame and slot detections.

        Raises:
            Exception: Propagates detector failures after logging traceback.
        """
        start_time = time.perf_counter()

        try:
            annotated_frame, detections, statistics = (
                self.slot_detector.process_frame(frame)
            )
        except Exception:
            logger.exception("Frame processing failed")
            raise

        statistics["fps"] = round(self._calculate_fps(start_time), 1)
        self.latest_statistics = self._normalize_statistics(statistics)
        return annotated_frame, detections

    def process_video(
        self,
        video_path: str | Path = VIDEO_PATH,
        output_path: str | Path = OUTPUT_PATH,
    ) -> None:
        """
        Process a video file and save annotated output.

        Args:
            video_path: Input video file path.
            output_path: Output video file path.

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

                try:
                    annotated_frame, _detections = self.process_frame(frame)
                except Exception:
                    continue

                writer.write(annotated_frame)
                if self.debug:
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
        Generate multipart MJPEG frames for Flask streaming.

        The video loops automatically when the end of the source is reached.

        Args:
            video_path: Input video file path.

        Yields:
            Multipart MJPEG-compatible frame bytes.

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

                try:
                    annotated_frame, _detections = self.process_frame(frame)
                except Exception:
                    continue

                try:
                    jpeg_bytes = self._encode_jpeg(annotated_frame)
                except Exception:
                    logger.exception("Frame encoding failed")
                    continue

                yield self._build_mjpeg_frame(jpeg_bytes)
        finally:
            capture.release()
            logger.info("Video released")

    @staticmethod
    def _initialize_detector() -> ParkingSlotYOLODetector:
        """
        Initialize the YOLO parking slot detector.

        Returns:
            Initialized ParkingSlotYOLODetector.

        Raises:
            RuntimeError: If detector initialization fails.
        """
        try:
            detector = ParkingSlotYOLODetector(model_path=SLOT_MODEL_PATH)
        except Exception:
            logger.exception("Detector initialization failed")
            raise

        logger.info("Detector initialized")
        return detector

    @staticmethod
    def _open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
        """
        Open a video source.

        Args:
            video_path: Input video path.

        Returns:
            Opened OpenCV VideoCapture.

        Raises:
            RuntimeError: If OpenCV cannot open the video.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video: {video_path}")

        logger.info("Video opened: %s", video_path)
        return capture

    @classmethod
    def _read_video_metadata(cls, capture: cv2.VideoCapture) -> VideoMetadata:
        """
        Read metadata from an opened video capture.

        Args:
            capture: Opened OpenCV VideoCapture.

        Returns:
            Source video metadata.

        Raises:
            RuntimeError: If the source video has invalid dimensions.
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
        Create a VideoWriter for annotated output.

        Args:
            output_path: Destination video path.
            metadata: Source video metadata.

        Returns:
            Opened OpenCV VideoWriter.

        Raises:
            RuntimeError: If OpenCV cannot create the writer.
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

    @classmethod
    def _encode_jpeg(cls, frame: np.ndarray) -> bytes:
        """
        Encode an annotated frame as JPEG.

        Args:
            frame: Annotated OpenCV frame.

        Returns:
            JPEG-encoded bytes.

        Raises:
            RuntimeError: If JPEG encoding fails.
        """
        success, buffer = cv2.imencode(cls.JPEG_EXTENSION, frame)
        if not success:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    @staticmethod
    def _restart_video(capture: cv2.VideoCapture) -> None:
        """Restart an opened video capture from the first frame."""
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info("Video restarted")

    @staticmethod
    def _calculate_fps(start_time: float) -> float:
        """
        Calculate instantaneous processing FPS.

        Args:
            start_time: Timestamp from time.perf_counter().

        Returns:
            Frames per second for the current frame.
        """
        elapsed = time.perf_counter() - start_time
        if elapsed <= 0:
            return 0.0
        return 1.0 / elapsed

    @staticmethod
    def _normalize_statistics(statistics: dict[str, Any]) -> Statistics:
        """
        Normalize detector statistics into the public statistics schema.

        Args:
            statistics: Raw statistics returned by the detector.

        Returns:
            Statistics dictionary compatible with app.py.
        """
        return {
            "total_slots": int(statistics.get("total_slots", 0)),
            "occupied_slots": int(statistics.get("occupied_slots", 0)),
            "available_slots": int(statistics.get("available_slots", 0)),
            "occupancy_percentage": float(
                statistics.get("occupancy_percentage", 0.0)
            ),
            "fps": float(statistics.get("fps", 0.0)),
        }

    @staticmethod
    def _build_mjpeg_frame(jpeg_bytes: bytes) -> bytes:
        """
        Build a multipart MJPEG frame payload.

        Args:
            jpeg_bytes: Encoded JPEG image bytes.

        Returns:
            Multipart frame bytes for Flask Response streaming.
        """
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )

    @staticmethod
    def _quit_requested() -> bool:
        """Return True when Q is pressed in debug preview mode."""
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), ord("Q"))

    def _release_resources(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter | None = None,
    ) -> None:
        """
        Release OpenCV resources.

        Args:
            capture: Video capture to release.
            writer: Optional video writer to release.
        """
        capture.release()
        if writer is not None:
            writer.release()
        if self.debug:
            cv2.destroyAllWindows()
        logger.info("Video released")


if __name__ == "__main__":
    pipeline = SmartParkingPipeline(debug=True)
    pipeline.process_video(VIDEO_PATH, OUTPUT_PATH)
