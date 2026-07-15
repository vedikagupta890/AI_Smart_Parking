"""Parking occupancy detection for the AI Smart Parking System."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import cv2
import numpy as np

BBox: TypeAlias = tuple[int, int, int, int]
VehicleDetection: TypeAlias = dict[str, Any]
OccupancyRecord: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class OccupancyConfig:
    """Configuration values for parking occupancy classification."""

    overlap_threshold: float = 0.15
    free_color: tuple[int, int, int] = (0, 255, 0)
    occupied_color: tuple[int, int, int] = (0, 0, 255)
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_background_color: tuple[int, int, int] = (0, 0, 0)
    rectangle_thickness: int = 2
    font_scale: float = 0.6
    summary_font_scale: float = 0.75
    text_thickness: int = 2


class OccupancyDetector:
    """Classify parking slots as free or occupied using vehicle overlaps."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, overlap_threshold: float = 0.15) -> None:
        """
        Initialize the occupancy detector.

        Args:
            overlap_threshold: Minimum fraction of a slot covered by a vehicle
                bbox before the slot is marked occupied.

        Raises:
            ValueError: If overlap_threshold is outside the range [0, 1].
        """
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1.")

        self.config = OccupancyConfig(overlap_threshold=overlap_threshold)

    def check_occupancy(
        self,
        vehicle_detections: list[VehicleDetection],
        parking_slots: list[BBox],
    ) -> list[OccupancyRecord]:
        """
        Classify each parking slot as occupied or free.

        Args:
            vehicle_detections: Vehicle detections from detector.py. Each
                detection must contain a "bbox" value as (x1, y1, x2, y2).
            parking_slots: Parking slot rectangles from slot_detector.py as
                (x1, y1, x2, y2).

        Returns:
            List of occupancy dictionaries:
            [{"id": 1, "occupied": True, "bbox": (...)}]
        """
        vehicle_boxes = self._extract_vehicle_boxes(vehicle_detections)
        occupancy_data: list[OccupancyRecord] = []

        for slot_id, slot_bbox in enumerate(parking_slots, start=1):
            normalized_slot = self._normalize_bbox(slot_bbox)
            occupied = self._is_slot_occupied(normalized_slot, vehicle_boxes)

            occupancy_data.append(
                {
                    "id": slot_id,
                    "occupied": occupied,
                    "bbox": normalized_slot,
                }
            )

        return occupancy_data

    def draw_occupancy(
        self,
        frame: np.ndarray,
        occupancy_data: list[OccupancyRecord],
    ) -> np.ndarray:
        """
        Draw parking occupancy state on a frame.

        Green rectangles represent free slots. Red rectangles represent
        occupied slots. The frame also displays available count, occupied
        count, and occupancy percentage.

        Args:
            frame: Input OpenCV frame.
            occupancy_data: Records returned by check_occupancy().

        Returns:
            Annotated frame.

        Raises:
            ValueError: If frame is not a valid OpenCV image.
        """
        self._validate_frame(frame)
        annotated_frame = frame.copy()

        for record in occupancy_data:
            self._draw_slot_status(annotated_frame, record)

        self._draw_summary(annotated_frame, occupancy_data)
        return annotated_frame

    def _is_slot_occupied(
        self,
        slot_bbox: BBox,
        vehicle_boxes: list[BBox],
    ) -> bool:
        """Return True when any vehicle overlaps a slot enough."""
        return any(
            self._slot_overlap_ratio(slot_bbox, vehicle_bbox)
            >= self.config.overlap_threshold
            for vehicle_bbox in vehicle_boxes
        )

    @staticmethod
    def _slot_overlap_ratio(slot_bbox: BBox, vehicle_bbox: BBox) -> float:
        """
        Calculate how much of a slot is covered by a vehicle box.

        The ratio is intersection area divided by slot area.
        """
        intersection_area = OccupancyDetector._intersection_area(
            slot_bbox,
            vehicle_bbox,
        )
        slot_area = OccupancyDetector._bbox_area(slot_bbox)

        if slot_area == 0:
            return 0.0

        return intersection_area / slot_area

    @staticmethod
    def _intersection_area(first_bbox: BBox, second_bbox: BBox) -> int:
        """Return intersection area for two bounding boxes."""
        first_x1, first_y1, first_x2, first_y2 = first_bbox
        second_x1, second_y1, second_x2, second_y2 = second_bbox

        x1 = max(first_x1, second_x1)
        y1 = max(first_y1, second_y1)
        x2 = min(first_x2, second_x2)
        y2 = min(first_y2, second_y2)

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height

    @staticmethod
    def _bbox_area(bbox: BBox) -> int:
        """Return area for a bounding box."""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _extract_vehicle_boxes(
        self,
        vehicle_detections: list[VehicleDetection],
    ) -> list[BBox]:
        """Extract normalized vehicle bounding boxes from detections."""
        vehicle_boxes: list[BBox] = []

        for detection in vehicle_detections:
            if "bbox" not in detection:
                continue

            vehicle_boxes.append(self._normalize_bbox(detection["bbox"]))

        return vehicle_boxes

    @staticmethod
    def _normalize_bbox(bbox: Any) -> BBox:
        """
        Convert a bbox-like object into a valid integer rectangle.

        Raises:
            ValueError: If bbox does not contain exactly four coordinates.
        """
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly four coordinates.")

        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        return left, top, right, bottom

    def _draw_slot_status(
        self,
        frame: np.ndarray,
        record: OccupancyRecord,
    ) -> None:
        """Draw one parking slot occupancy record."""
        x1, y1, x2, y2 = self._normalize_bbox(record["bbox"])
        occupied = bool(record["occupied"])
        color = (
            self.config.occupied_color
            if occupied
            else self.config.free_color
        )
        label = f"Slot {record['id']}: {'Occupied' if occupied else 'Free'}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self.config.rectangle_thickness,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(20, y1 - 8)),
            self.FONT,
            self.config.font_scale,
            color,
            self.config.text_thickness,
            cv2.LINE_AA,
        )

    def _draw_summary(
        self,
        frame: np.ndarray,
        occupancy_data: list[OccupancyRecord],
    ) -> None:
        """Draw available, occupied, and occupancy percentage summary."""
        total_slots = len(occupancy_data)
        occupied_count = sum(
            1
            for record in occupancy_data
            if bool(record["occupied"])
        )
        available_count = total_slots - occupied_count
        occupancy_percent = (
            (occupied_count / total_slots) * 100.0
            if total_slots
            else 0.0
        )

        summary_lines = [
            f"Available: {available_count}",
            f"Occupied: {occupied_count}",
            f"Occupancy: {occupancy_percent:.1f}%",
        ]

        self._draw_text_block(frame, summary_lines, origin=(15, 30))

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
    ) -> None:
        """Draw a readable text block on the frame."""
        x, y = origin
        line_height = 28
        padding = 8
        max_width = 0

        for line in lines:
            text_size, _ = cv2.getTextSize(
                line,
                self.FONT,
                self.config.summary_font_scale,
                self.config.text_thickness,
            )
            max_width = max(max_width, text_size[0])

        block_height = line_height * len(lines) + padding
        cv2.rectangle(
            frame,
            (x - padding, y - 22),
            (x + max_width + padding, y - 22 + block_height),
            self.config.text_background_color,
            thickness=-1,
        )

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + index * line_height),
                self.FONT,
                self.config.summary_font_scale,
                self.config.text_color,
                self.config.text_thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate that frame is a non-empty OpenCV-compatible image."""
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array.")

        if frame.size == 0:
            raise ValueError("frame must not be empty.")

        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2D or 3D image array.")
