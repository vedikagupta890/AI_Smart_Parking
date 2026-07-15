"""Parking slot candidate detection using classical computer vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

Line: TypeAlias = tuple[int, int, int, int]
Slot: TypeAlias = tuple[int, int, int, int]


@dataclass(frozen=True)
class SlotDetectorConfig:
    """Tunable parameters for the parking slot detection pipeline."""

    blur_kernel_size: tuple[int, int] = (5, 5)
    blur_sigma: float = 0.0
    canny_threshold_low: int = 50
    canny_threshold_high: int = 150
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180.0
    hough_threshold: int = 45
    hough_min_line_length_ratio: float = 0.08
    hough_max_line_gap_ratio: float = 0.025
    min_hough_line_length: int = 45
    min_hough_line_gap: int = 12
    separator_angle_tolerance_degrees: float = 25.0
    line_group_distance_ratio: float = 0.025
    min_line_group_distance: int = 18
    min_slot_width_ratio: float = 0.035
    max_slot_width_ratio: float = 0.25
    min_slot_height_ratio: float = 0.08
    slot_padding_ratio: float = 0.006
    duplicate_iou_threshold: float = 0.75
    rectangle_thickness: int = 2
    label_font_scale: float = 0.55
    label_thickness: int = 2
    summary_font_scale: float = 0.8
    summary_thickness: int = 2


@dataclass(frozen=True)
class FrameMetrics:
    """Frame dimensions used for resolution-aware thresholds."""

    height: int
    width: int

    @property
    def short_side(self) -> int:
        """Return the smaller frame dimension."""
        return min(self.height, self.width)


@dataclass(frozen=True)
class LineGroup:
    """Representative geometry for a cluster of nearby separator lines."""

    x: int
    y_min: int
    y_max: int


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata required to create an output video writer."""

    fps: float
    width: int
    height: int


class ParkingSlotDetector:
    """Detect likely parking slot rectangles from parking lot frames."""

    WINDOW_NAME = "Parking Slot Detection"
    OUTPUT_CODEC = "mp4v"
    SLOT_COLOR: tuple[int, int, int] = (0, 255, 0)
    TEXT_COLOR: tuple[int, int, int] = (0, 255, 0)
    TEXT_BACKGROUND_COLOR: tuple[int, int, int] = (0, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the parking slot detector.

        Args:
            debug: If True, keep copies of intermediate images from the most
                recent processed frame in debug_images.
        """
        self.debug = debug
        self.config = SlotDetectorConfig()
        self.debug_images: dict[str, np.ndarray] = {}
        self._frame_metrics: FrameMetrics | None = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a frame to grayscale and apply Gaussian blur.

        Args:
            frame: Input OpenCV frame in BGR or grayscale format.

        Returns:
            Blurred grayscale image.

        Raises:
            ValueError: If frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        self._set_frame_metrics(frame)

        grayscale = self._to_grayscale(frame)
        processed_image = cv2.GaussianBlur(
            grayscale,
            self.config.blur_kernel_size,
            self.config.blur_sigma,
        )

        self._store_debug_image("preprocessed", processed_image)
        return processed_image

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection.

        Args:
            image: Preprocessed grayscale image.

        Returns:
            Binary edge image.

        Raises:
            ValueError: If image is not a valid image array.
        """
        self._validate_image(image, "image")

        edge_image = cv2.Canny(
            image,
            self.config.canny_threshold_low,
            self.config.canny_threshold_high,
        )

        self._store_debug_image("edges", edge_image)
        return edge_image

    def detect_lines(self, edge_image: np.ndarray) -> list[Line]:
        """
        Detect line segments using Probabilistic Hough Transform.

        Args:
            edge_image: Binary edge image returned by detect_edges().

        Returns:
            Detected lines as (x1, y1, x2, y2).

        Raises:
            ValueError: If edge_image is not a valid image array.
        """
        self._validate_image(edge_image, "edge_image")
        metrics = self._metrics_from_image(edge_image)
        min_line_length, max_line_gap = self._hough_line_parameters(metrics)

        raw_lines = cv2.HoughLinesP(
            edge_image,
            rho=self.config.hough_rho,
            theta=self.config.hough_theta,
            threshold=self.config.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        if raw_lines is None:
            return []

        return self._normalize_hough_lines(raw_lines)

    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the region of interest for parking slot detection.

        This currently returns the full frame. The method is isolated so a
        polygon mask, crop box, or perspective transform can be added later
        without changing callers.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Region of interest image.

        Raises:
            ValueError: If frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        self._set_frame_metrics(frame)

        roi = frame.copy()
        self._store_debug_image("roi", roi)
        return roi

    def generate_slots(self, lines: list[Line]) -> list[Slot]:
        """
        Estimate rectangular parking slot candidates from detected lines.

        Nearby near-parallel separator lines are grouped by horizontal
        position, then adjacent line groups are paired into slot rectangles.

        Args:
            lines: Hough line segments as (x1, y1, x2, y2).

        Returns:
            Slot rectangles as (x1, y1, x2, y2).
        """
        if not lines:
            return []

        metrics = self._get_frame_metrics(lines)
        separator_lines = self._filter_separator_lines(lines)
        line_groups = self._group_parallel_lines(separator_lines, metrics)
        slot_candidates = self._build_slot_candidates(line_groups, metrics)
        return self._deduplicate_slots(slot_candidates, metrics)

    def detect_slots(self, frame: np.ndarray) -> list[Slot]:
        """
        Execute the full parking slot detection pipeline.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Detected parking slot rectangles.
        """
        self._clear_debug_images()
        roi = self.extract_roi(frame)
        processed_image = self.preprocess(roi)
        edge_image = self.detect_edges(processed_image)
        lines = self.detect_lines(edge_image)
        return self.generate_slots(lines)

    def draw_slots(self, frame: np.ndarray, slots: list[Slot]) -> np.ndarray:
        """
        Draw detected parking slots and slot count on a frame.

        Args:
            frame: Input OpenCV frame.
            slots: Slot rectangles as (x1, y1, x2, y2).

        Returns:
            Annotated frame.

        Raises:
            ValueError: If frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        annotated_frame = frame.copy()

        for slot_id, slot in enumerate(slots, start=1):
            self._draw_slot(annotated_frame, slot, slot_id)

        self._draw_slot_count(annotated_frame, len(slots))
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Slot]]:
        """
        Process a single frame and return annotation plus structured slots.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Tuple containing annotated_frame and slot_list.
        """
        slot_list = self.detect_slots(frame)
        annotated_frame = self.draw_slots(frame, slot_list)
        return annotated_frame, slot_list

    def process_video(self, video_path: str | Path, output_path: str | Path) -> None:
        """
        Process a video, save annotated output, and display live results.

        Press q to exit early.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the annotated output video.

        Raises:
            FileNotFoundError: If the input video file does not exist.
            OSError: If the video or output writer cannot be opened.
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
        """Process frames from an open capture and write annotations."""
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotated_frame, _ = self.process_frame(frame)
            writer.write(annotated_frame)
            cv2.imshow(self.WINDOW_NAME, annotated_frame)

            if self._quit_requested():
                break

    def _build_slot_candidates(
        self,
        line_groups: list[LineGroup],
        metrics: FrameMetrics,
    ) -> list[Slot]:
        """Create slot candidates from adjacent separator line groups."""
        if len(line_groups) < 2:
            return []

        min_width, max_width, min_height = self._slot_size_limits(metrics)
        padding = max(2, int(metrics.short_side * self.config.slot_padding_ratio))
        slots: list[Slot] = []

        for left_group, right_group in zip(line_groups, line_groups[1:]):
            slot = self._slot_from_line_pair(
                left_group,
                right_group,
                metrics,
                padding,
            )

            if self._is_valid_slot(slot, metrics, min_width, max_width, min_height):
                slots.append(slot)

        return slots

    def _slot_from_line_pair(
        self,
        left_group: LineGroup,
        right_group: LineGroup,
        metrics: FrameMetrics,
        padding: int,
    ) -> Slot:
        """Estimate an axis-aligned slot rectangle from two line groups."""
        y1 = max(0, min(left_group.y_min, right_group.y_min) - padding)
        y2 = min(
            metrics.height - 1,
            max(left_group.y_max, right_group.y_max) + padding,
        )
        return left_group.x, y1, right_group.x, y2

    def _slot_size_limits(self, metrics: FrameMetrics) -> tuple[int, int, int]:
        """Return minimum width, maximum width, and minimum height."""
        min_width = max(12, int(metrics.width * self.config.min_slot_width_ratio))
        max_width = max(
            min_width + 1,
            int(metrics.width * self.config.max_slot_width_ratio),
        )
        min_height = max(20, int(metrics.height * self.config.min_slot_height_ratio))
        return min_width, max_width, min_height

    @staticmethod
    def _is_valid_slot(
        slot: Slot,
        metrics: FrameMetrics,
        min_width: int,
        max_width: int,
        min_height: int,
    ) -> bool:
        """Return True when a slot has plausible dimensions and position."""
        x1, y1, x2, y2 = slot
        width = x2 - x1
        height = y2 - y1

        return (
            0 <= x1 < x2 < metrics.width
            and 0 <= y1 < y2 < metrics.height
            and min_width <= width <= max_width
            and height >= min_height
        )

    def _filter_separator_lines(self, lines: list[Line]) -> list[Line]:
        """Return lines likely to represent parking slot separators."""
        return [line for line in lines if self._is_separator_line(line)]

    def _is_separator_line(self, line: Line) -> bool:
        """Return True for near-vertical separator lines."""
        angle = abs(self._line_angle_degrees(line))
        distance_from_vertical = abs(90.0 - angle)
        return distance_from_vertical <= (
            self.config.separator_angle_tolerance_degrees
        )

    def _group_parallel_lines(
        self,
        lines: list[Line],
        metrics: FrameMetrics,
    ) -> list[LineGroup]:
        """Group nearby separator lines into representative parallel lines."""
        if not lines:
            return []

        group_distance = self._line_group_distance(metrics)
        sorted_lines = sorted(lines, key=self._line_center_x)
        line_groups: list[list[Line]] = [[sorted_lines[0]]]
        group_centers = [float(self._line_center_x(sorted_lines[0]))]

        for line in sorted_lines[1:]:
            center_x = self._line_center_x(line)

            if abs(center_x - group_centers[-1]) <= group_distance:
                line_groups[-1].append(line)
                group_centers[-1] = self._mean_line_center(line_groups[-1])
            else:
                line_groups.append([line])
                group_centers.append(float(center_x))

        return [self._line_group_from_lines(group) for group in line_groups]

    def _line_group_distance(self, metrics: FrameMetrics) -> int:
        """Return distance threshold for grouping nearby lines."""
        return max(
            self.config.min_line_group_distance,
            int(metrics.width * self.config.line_group_distance_ratio),
        )

    @staticmethod
    def _line_group_from_lines(lines: list[Line]) -> LineGroup:
        """Create one representative line group from clustered lines."""
        x_values = [
            coordinate
            for x1, _, x2, _ in lines
            for coordinate in (x1, x2)
        ]
        y_values = [
            coordinate
            for _, y1, _, y2 in lines
            for coordinate in (y1, y2)
        ]

        return LineGroup(
            x=int(round(float(np.mean(x_values)))),
            y_min=int(min(y_values)),
            y_max=int(max(y_values)),
        )

    def _deduplicate_slots(
        self,
        slots: list[Slot],
        metrics: FrameMetrics,
    ) -> list[Slot]:
        """Remove duplicate or heavily overlapping slot rectangles."""
        unique_slots: list[Slot] = []
        sorted_slots = sorted(
            slots,
            key=lambda slot: (slot[0], slot[1], slot[2], slot[3]),
        )

        for slot in sorted_slots:
            if self._matches_existing_slot(slot, unique_slots, metrics):
                continue
            unique_slots.append(slot)

        return unique_slots

    def _matches_existing_slot(
        self,
        slot: Slot,
        existing_slots: list[Slot],
        metrics: FrameMetrics,
    ) -> bool:
        """Return True if a slot duplicates any accepted slot."""
        return any(
            self._slot_iou(slot, existing) >= self.config.duplicate_iou_threshold
            or self._slots_have_near_equal_edges(slot, existing, metrics)
            for existing in existing_slots
        )

    def _slots_have_near_equal_edges(
        self,
        first_slot: Slot,
        second_slot: Slot,
        metrics: FrameMetrics,
    ) -> bool:
        """Return True when two slots have almost identical edges."""
        tolerance = self._line_group_distance(metrics)
        return all(
            abs(first - second) <= tolerance
            for first, second in zip(first_slot, second_slot)
        )

    @staticmethod
    def _slot_iou(first_slot: Slot, second_slot: Slot) -> float:
        """Calculate intersection-over-union for two slot rectangles."""
        first_x1, first_y1, first_x2, first_y2 = first_slot
        second_x1, second_y1, second_x2, second_y2 = second_slot

        intersection_x1 = max(first_x1, second_x1)
        intersection_y1 = max(first_y1, second_y1)
        intersection_x2 = min(first_x2, second_x2)
        intersection_y2 = min(first_y2, second_y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        if intersection_area == 0:
            return 0.0

        first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
        second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
        union_area = first_area + second_area - intersection_area

        return intersection_area / union_area if union_area else 0.0

    def _draw_slot(self, frame: np.ndarray, slot: Slot, slot_id: int) -> None:
        """Draw one labeled parking slot rectangle."""
        x1, y1, x2, y2 = slot
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.SLOT_COLOR,
            self.config.rectangle_thickness,
        )
        cv2.putText(
            frame,
            f"Slot {slot_id}",
            (x1 + 4, max(20, y1 - 8)),
            self.FONT,
            self.config.label_font_scale,
            self.TEXT_COLOR,
            self.config.label_thickness,
            cv2.LINE_AA,
        )

    def _draw_slot_count(self, frame: np.ndarray, slot_count: int) -> None:
        """Draw total detected slot count at the top-left corner."""
        text = f"Slots: {slot_count}"
        text_size, baseline = cv2.getTextSize(
            text,
            self.FONT,
            self.config.summary_font_scale,
            self.config.summary_thickness,
        )
        text_width, text_height = text_size
        x, y = 15, 35

        cv2.rectangle(
            frame,
            (x - 8, y - text_height - baseline - 8),
            (x + text_width + 8, y + baseline + 8),
            self.TEXT_BACKGROUND_COLOR,
            thickness=-1,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            self.FONT,
            self.config.summary_font_scale,
            self.TEXT_COLOR,
            self.config.summary_thickness,
            cv2.LINE_AA,
        )

    def _hough_line_parameters(self, metrics: FrameMetrics) -> tuple[int, int]:
        """Return resolution-aware Hough minLineLength and maxLineGap."""
        min_line_length = max(
            self.config.min_hough_line_length,
            int(metrics.short_side * self.config.hough_min_line_length_ratio),
        )
        max_line_gap = max(
            self.config.min_hough_line_gap,
            int(metrics.short_side * self.config.hough_max_line_gap_ratio),
        )
        return min_line_length, max_line_gap

    @staticmethod
    def _normalize_hough_lines(raw_lines: np.ndarray) -> list[Line]:
        """Convert OpenCV Hough output into Python integer tuples."""
        return [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4)
        ]

    @staticmethod
    def _line_angle_degrees(line: Line) -> float:
        """Return a line segment angle in degrees."""
        x1, y1, x2, y2 = line
        return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    @staticmethod
    def _line_center_x(line: Line) -> int:
        """Return the horizontal midpoint of a line segment."""
        x1, _, x2, _ = line
        return int(round((x1 + x2) / 2.0))

    @staticmethod
    def _mean_line_center(lines: list[Line]) -> float:
        """Return the mean horizontal midpoint for a list of lines."""
        centers = [ParkingSlotDetector._line_center_x(line) for line in lines]
        return float(np.mean(centers))

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Return a grayscale image from BGR or grayscale input."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _set_frame_metrics(self, image: np.ndarray) -> None:
        """Store frame dimensions for later calculations."""
        self._frame_metrics = self._metrics_from_image(image)

    @staticmethod
    def _metrics_from_image(image: np.ndarray) -> FrameMetrics:
        """Return FrameMetrics from an image array."""
        height, width = image.shape[:2]
        return FrameMetrics(height=height, width=width)

    def _get_frame_metrics(self, lines: list[Line]) -> FrameMetrics:
        """Return known frame metrics or infer dimensions from lines."""
        if self._frame_metrics is not None:
            return self._frame_metrics

        max_x = max(max(x1, x2) for x1, _, x2, _ in lines)
        max_y = max(max(y1, y2) for _, y1, _, y2 in lines)
        return FrameMetrics(height=max_y + 1, width=max_x + 1)

    @staticmethod
    def _read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
        """Read metadata from an open VideoCapture."""
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
        """Create an output VideoWriter matching the input video."""
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
        """Return True when the user presses q."""
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> None:
        """Validate that an input is a non-empty OpenCV image array."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")

        if image.size == 0:
            raise ValueError(f"{name} must not be empty.")

        if image.ndim not in (2, 3):
            raise ValueError(f"{name} must be a 2D or 3D image array.")

    def _store_debug_image(self, name: str, image: np.ndarray) -> None:
        """Store an intermediate image only when debug mode is enabled."""
        if self.debug:
            self.debug_images[name] = image.copy()

    def _clear_debug_images(self) -> None:
        """Clear previous debug images only when debug mode is enabled."""
        if self.debug:
            self.debug_images.clear()


if __name__ == "__main__":
    detector = ParkingSlotDetector(debug=False)
    detector.process_video(
        "data/videos/parking.mp4",
        "output/slot_detection_result.mp4",
    )
