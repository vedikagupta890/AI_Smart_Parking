"""End-to-end video pipeline for the AI Smart Parking System."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ULTRALYTICS_CONFIG_DIR = PROJECT_DIR / ".ultralytics"


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


class SmartParkingPipeline:
    """Coordinate vehicle detection, slot detection, and occupancy analysis."""

    WINDOW_NAME = "AI Smart Parking System"
    OUTPUT_CODEC = "mp4v"

    def __init__(
        self,
        vehicle_detector: VehicleDetector | None = None,
        slot_detector: ParkingSlotDetector | None = None,
        occupancy_detector: OccupancyDetector | None = None,
    ) -> None:
        """
        Initialize the smart parking pipeline.

        Args:
            vehicle_detector: Optional preconfigured vehicle detector.
            slot_detector: Optional preconfigured parking slot detector.
            occupancy_detector: Optional preconfigured occupancy detector.
        """
        self.vehicle_detector = vehicle_detector or VehicleDetector()
        self.slot_detector = slot_detector or ParkingSlotDetector()
        self.occupancy_detector = occupancy_detector or OccupancyDetector()

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
        """
        vehicle_detections = self.vehicle_detector.detect(frame)
        parking_slots = self.slot_detector.detect_slots(frame)
        occupancy_data = self.occupancy_detector.check_occupancy(
            vehicle_detections,
            parking_slots,
        )
        annotated_frame = self.occupancy_detector.draw_occupancy(
            frame,
            occupancy_data,
        )

        return annotated_frame, occupancy_data

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

        metadata = self._read_video_metadata(capture)
        writer = self._create_video_writer(output_file, metadata)

        try:
            self._process_video_stream(capture, writer)
        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()

    def _process_video_stream(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter,
    ) -> None:
        """Read, process, display, and save frames until video end or Q."""
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated_frame, _ = self.process_frame(frame)
            writer.write(annotated_frame)
            cv2.imshow(self.WINDOW_NAME, annotated_frame)

            if self._quit_requested():
                break

    @staticmethod
    def _read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
        """
        Read and validate video metadata from an open capture object.

        Args:
            capture: OpenCV video capture object.

        Returns:
            VideoMetadata containing fps, width, and height.

        Raises:
            OSError: If frame dimensions cannot be read.
        """
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise OSError("Unable to read input video dimensions.")

        if fps <= 0:
            fps = 30.0

        return VideoMetadata(fps=fps, width=width, height=height)

    def _create_video_writer(
        self,
        output_path: Path,
        metadata: VideoMetadata,
    ) -> cv2.VideoWriter:
        """
        Create a VideoWriter using input video metadata.

        Args:
            output_path: Output file path.
            metadata: Input video metadata.

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
    def _quit_requested() -> bool:
        """Return True when the user presses Q."""
        return (cv2.waitKey(1) & 0xFF) == ord("q")


def main() -> None:
    """Run the default smart parking video pipeline."""
    pipeline = SmartParkingPipeline()
    pipeline.process_video(
        "data/videos/parking.mp4",
        "output/parking_result.mp4",
    )


if __name__ == "__main__":
    main()
