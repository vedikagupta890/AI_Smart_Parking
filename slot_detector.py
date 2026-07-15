"""Parking slot candidate detection for the AI Smart Parking System."""

from __future__ import annotations

from dataclasses import dataclass
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
    vertical_angle_tolerance_degrees: float = 25.0
    line_group_distance_ratio: float = 0.025
    min_line_group_distance: int = 18
    min_slot_width_ratio: float = 0.035
    max_slot_width_ratio: float = 0.22
    min_slot_height_ratio: float = 0.08
    slot_padding_ratio: float = 0.006
    duplicate_iou_threshold: float = 0.75
    label_font_scale: float = 0.55
    label_thickness: int = 2
    rectangle_thickness: int = 2


@dataclass(frozen=True)
class LineGroup:
    """Representative geometry for clustered slot separator lines."""

    x: int
    y_min: int
    y_max: int


@dataclass(frozen=True)
class FrameMetrics:
    """Frame-specific dimensions and derived geometry thresholds."""

    height: int
    width: int

    @property
    def short_side(self) -> int:
        """Return the smaller frame dimension."""
        return min(self.height, self.width)


class ParkingSlotDetector:
    """Detect likely parking slot rectangles from parking lot frames."""

    SLOT_COLOR: tuple[int, int, int] = (0, 255, 0)
    LABEL_COLOR: tuple[int, int, int] = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the parking slot detector.

        Args:
            debug: Whether to store intermediate pipeline images for
                inspection during development.
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
            ValueError: If the frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        self._set_frame_metrics(frame)

        grayscale = self._to_grayscale(frame)
        processed = cv2.GaussianBlur(
            grayscale,
            self.config.blur_kernel_size,
            self.config.blur_sigma,
        )

        self._store_debug_image("preprocessed", processed)
        return processed

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.

        Args:
            image: Grayscale or preprocessed image.

        Returns:
            Binary edge image.

        Raises:
            ValueError: If the image is not a valid image array.
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

        Hough parameters are scaled to the current frame size so the detector
        behaves more consistently across common video resolutions.

        Args:
            edge_image: Binary edge image returned by detect_edges().

        Returns:
            List of detected line segments as (x1, y1, x2, y2).

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
        Extract the parking area from a frame.

        This currently returns a copy of the original frame. Keeping ROI
        extraction in a separate method allows polygon masks, crop boxes, or
        perspective transforms to be added later without changing callers.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Region of interest image.

        Raises:
            ValueError: If the frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        self._set_frame_metrics(frame)

        roi = frame.copy()
        self._store_debug_image("roi", roi)
        return roi

    def generate_slots(self, lines: list[Line]) -> list[Slot]:
        """
        Generate parking slot rectangles from detected line segments.

        Nearby near-vertical lines are grouped by horizontal position.
        Adjacent line groups are paired to estimate slot rectangles.

        Args:
            lines: Line segments as (x1, y1, x2, y2).

        Returns:
            List of unique slot rectangles as (x1, y1, x2, y2).
        """
        if not lines:
            return []

        metrics = self._get_frame_metrics(lines)
        vertical_lines = self._filter_vertical_lines(lines)
        line_groups = self._group_vertical_lines(vertical_lines, metrics)
        slot_candidates = self._build_slot_candidates(line_groups, metrics)
        return self._deduplicate_slots(slot_candidates, metrics)

    def detect_slots(self, frame: np.ndarray) -> list[Slot]:
        """
        Run the full parking slot detection pipeline.

        Args:
            frame: Input OpenCV frame.

        Returns:
            List of detected slot rectangles as (x1, y1, x2, y2).
        """
        roi = self.extract_roi(frame)
        processed = self.preprocess(roi)
        edge_image = self.detect_edges(processed)
        lines = self.detect_lines(edge_image)
        return self.generate_slots(lines)

    def draw_slots(self, frame: np.ndarray, slots: list[Slot]) -> np.ndarray:
        """
        Draw parking slot rectangles and labels on a frame.

        Args:
            frame: Input OpenCV frame.
            slots: Slot rectangles as (x1, y1, x2, y2).

        Returns:
            Annotated frame.

        Raises:
            ValueError: If the frame is not a valid image array.
        """
        self._validate_image(frame, "frame")
        annotated_frame = frame.copy()

        for index, slot in enumerate(slots, start=1):
            self._draw_slot(annotated_frame, slot, index)

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Slot]]:
        """
        Detect parking slots and return both annotation and structured data.

        Args:
            frame: Input OpenCV frame.

        Returns:
            Tuple containing the annotated frame and slot rectangle list.
        """
        slots = self.detect_slots(frame)
        annotated_frame = self.draw_slots(frame, slots)
        return annotated_frame, slots

    def _build_slot_candidates(
        self,
        line_groups: list[LineGroup],
        metrics: FrameMetrics,
    ) -> list[Slot]:
        """Create slot rectangles from adjacent vertical line groups."""
        if len(line_groups) < 2:
            return []

        min_width = max(12, int(metrics.width * self.config.min_slot_width_ratio))
        max_width = max(
            min_width + 1,
            int(metrics.width * self.config.max_slot_width_ratio),
        )
        min_height = max(20, int(metrics.height * self.config.min_slot_height_ratio))
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
        """Estimate a rectangle between two adjacent separator lines."""
        y1 = max(0, min(left_group.y_min, right_group.y_min) - padding)
        y2 = min(
            metrics.height - 1,
            max(left_group.y_max, right_group.y_max) + padding,
        )
        return left_group.x, y1, right_group.x, y2

    @staticmethod
    def _is_valid_slot(
        slot: Slot,
        metrics: FrameMetrics,
        min_width: int,
        max_width: int,
        min_height: int,
    ) -> bool:
        """Return True when a slot rectangle has plausible dimensions."""
        x1, y1, x2, y2 = slot
        slot_width = x2 - x1
        slot_height = y2 - y1

        return (
            0 <= x1 < x2 < metrics.width
            and 0 <= y1 < y2 < metrics.height
            and min_width <= slot_width <= max_width
            and slot_height >= min_height
        )

    def _filter_vertical_lines(self, lines: list[Line]) -> list[Line]:
        """Return only lines close enough to vertical parking separators."""
        return [line for line in lines if self._is_vertical_line(line)]

    def _group_vertical_lines(
        self,
        lines: list[Line],
        metrics: FrameMetrics,
    ) -> list[LineGroup]:
        """Cluster nearby vertical lines into representative separators."""
        if not lines:
            return []

        group_distance = max(
            self.config.min_line_group_distance,
            int(metrics.width * self.config.line_group_distance_ratio),
        )
        sorted_lines = sorted(lines, key=self._line_center_x)
        grouped_lines: list[list[Line]] = [[sorted_lines[0]]]
        grouped_centers = [float(self._line_center_x(sorted_lines[0]))]

        for line in sorted_lines[1:]:
            center_x = self._line_center_x(line)

            if abs(center_x - grouped_centers[-1]) <= group_distance:
                grouped_lines[-1].append(line)
                grouped_centers[-1] = self._mean_line_center(grouped_lines[-1])
            else:
                grouped_lines.append([line])
                grouped_centers.append(float(center_x))

        return [self._line_group_from_lines(group) for group in grouped_lines]

    @staticmethod
    def _line_group_from_lines(lines: list[Line]) -> LineGroup:
        """Build one representative separator from a cluster of lines."""
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
        """Remove duplicated or heavily overlapping slot rectangles."""
        unique_slots: list[Slot] = []

        sorted_slots = sorted(
            slots,
            key=lambda item: (item[0], item[1], item[2], item[3]),
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
        """Return True when a slot duplicates an accepted rectangle."""
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
        """Return True when two slots have nearly identical rectangle edges."""
        tolerance = max(
            self.config.min_line_group_distance,
            int(metrics.width * self.config.line_group_distance_ratio),
        )
        return all(
            abs(first - second) <= tolerance
            for first, second in zip(first_slot, second_slot)
        )

    @staticmethod
    def _slot_iou(first_slot: Slot, second_slot: Slot) -> float:
        """Compute intersection-over-union for two slot rectangles."""
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

    def _draw_slot(
        self,
        frame: np.ndarray,
        slot: Slot,
        index: int,
    ) -> None:
        """Draw a single labeled parking slot on a frame."""
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
            f"Slot {index}",
            (x1 + 4, max(18, y1 - 8)),
            self.FONT,
            self.config.label_font_scale,
            self.LABEL_COLOR,
            self.config.label_thickness,
            cv2.LINE_AA,
        )

    def _is_vertical_line(self, line: Line) -> bool:
        """Return True when a line is close enough to vertical."""
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            return True

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        distance_from_vertical = abs(90.0 - angle)
        return distance_from_vertical <= (
            self.config.vertical_angle_tolerance_degrees
        )

    def _hough_line_parameters(self, metrics: FrameMetrics) -> tuple[int, int]:
        """Return resolution-aware Hough line length and gap settings."""
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
        """Convert OpenCV Hough output into plain Python integer tuples."""
        return [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in raw_lines.reshape(-1, 4)
        ]

    @staticmethod
    def _mean_line_center(lines: list[Line]) -> float:
        """Return the mean horizontal midpoint for a group of lines."""
        centers = [
            ParkingSlotDetector._line_center_x(line)
            for line in lines
        ]
        return float(np.mean(centers))

    @staticmethod
    def _line_center_x(line: Line) -> int:
        """Return the horizontal midpoint of a line segment."""
        x1, _, x2, _ = line
        return int(round((x1 + x2) / 2.0))

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Return a grayscale view or conversion of an OpenCV image."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _set_frame_metrics(self, image: np.ndarray) -> None:
        """Store dimensions for later resolution-aware calculations."""
        self._frame_metrics = self._metrics_from_image(image)

    @staticmethod
    def _metrics_from_image(image: np.ndarray) -> FrameMetrics:
        """Return frame metrics from an image array."""
        height, width = image.shape[:2]
        return FrameMetrics(height=height, width=width)

    def _get_frame_metrics(self, lines: list[Line]) -> FrameMetrics:
        """Return stored frame metrics, falling back to line extents."""
        if self._frame_metrics is not None:
            return self._frame_metrics

        max_x = max(max(x1, x2) for x1, _, x2, _ in lines)
        max_y = max(max(y1, y2) for _, y1, _, y2 in lines)
        return FrameMetrics(height=max_y + 1, width=max_x + 1)

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> None:
        """Validate that an input is a non-empty OpenCV-compatible image."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")

        if image.size == 0:
            raise ValueError(f"{name} must not be empty.")

        if image.ndim not in (2, 3):
            raise ValueError(f"{name} must be a 2D or 3D image array.")

    def _store_debug_image(self, name: str, image: np.ndarray) -> None:
        """Store a copy of an intermediate image when debug mode is enabled."""
        if self.debug:
            self.debug_images[name] = image.copy()
