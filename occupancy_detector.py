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


@dataclass(frozen=True)
class VehicleMatch:
    """Normalized vehicle data used for occupancy matching."""

    bbox: BBox
    class_name: str | None
    confidence: float | None


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
            [
                {
                    "id": 1,
                    "bbox": (...),
                    "occupied": True,
                    "vehicle_class": "car",
                    "vehicle_confidence": 0.94,
                    "overlap_score": 0.82,
                }
            ]
        """
        vehicle_matches = self._extract_vehicle_matches(vehicle_detections)
        occupancy_data: list[OccupancyRecord] = []
        next_slot_id = 1

        for slot_bbox in self._safe_iterable(parking_slots):
            normalized_slot = self._try_normalize_bbox(slot_bbox)
            if normalized_slot is None:
                continue

            best_match, overlap_score = self._find_best_vehicle_match(
                normalized_slot,
                vehicle_matches,
            )
            occupied = (
                best_match is not None
                and overlap_score >= self.config.overlap_threshold
            )

            occupancy_data.append(
                {
                    "id": next_slot_id,
                    "bbox": normalized_slot,
                    "occupied": occupied,
                    "vehicle_class": (
                        best_match.class_name
                        if occupied and best_match is not None
                        else None
                    ),
                    "vehicle_confidence": (
                        best_match.confidence
                        if occupied and best_match is not None
                        else None
                    ),
                    "overlap_score": overlap_score if occupied else 0.0,
                }
            )
            next_slot_id += 1

        return occupancy_data

    def get_statistics(
        self,
        occupancy_data: list[OccupancyRecord],
    ) -> dict[str, int | float]:
        """
        Calculate parking occupancy statistics.

        Args:
            occupancy_data: Records returned by check_occupancy().

        Returns:
            Dictionary with total, occupied, available, and percentage values.
        """
        valid_records = [
            record
            for record in self._safe_iterable(occupancy_data)
            if isinstance(record, dict)
            if self._try_normalize_bbox(record.get("bbox")) is not None
        ]
        total_slots = len(valid_records)
        occupied_slots = sum(
            1
            for record in valid_records
            if bool(record.get("occupied", False))
        )
        available_slots = total_slots - occupied_slots
        occupancy_percentage = (
            (occupied_slots / total_slots) * 100.0
            if total_slots
            else 0.0
        )

        return {
            "total_slots": total_slots,
            "occupied_slots": occupied_slots,
            "available_slots": available_slots,
            "occupancy_percentage": occupancy_percentage,
        }

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

        for record in self._safe_iterable(occupancy_data):
            if not isinstance(record, dict):
                continue
            self._draw_slot_status(annotated_frame, record)

        self._draw_summary(annotated_frame, occupancy_data)
        return annotated_frame

    def _find_best_vehicle_match(
        self,
        slot_bbox: BBox,
        vehicle_matches: list[VehicleMatch],
    ) -> tuple[VehicleMatch | None, float]:
        """Return the vehicle with the highest slot overlap."""
        best_match: VehicleMatch | None = None
        best_overlap = 0.0

        for vehicle_match in vehicle_matches:
            overlap_score = self._slot_overlap_ratio(
                slot_bbox,
                vehicle_match.bbox,
            )
            if overlap_score > best_overlap:
                best_match = vehicle_match
                best_overlap = overlap_score

        return best_match, best_overlap

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

    def _extract_vehicle_matches(
        self,
        vehicle_detections: list[VehicleDetection],
    ) -> list[VehicleMatch]:
        """Extract normalized vehicle match data from detections."""
        vehicle_matches: list[VehicleMatch] = []

        for detection in self._safe_iterable(vehicle_detections):
            if not isinstance(detection, dict):
                continue

            bbox = self._try_normalize_bbox(detection.get("bbox"))
            if bbox is None:
                continue

            vehicle_matches.append(
                VehicleMatch(
                    bbox=bbox,
                    class_name=self._optional_string(
                        detection.get("class_name")
                    ),
                    confidence=self._optional_float(
                        detection.get("confidence")
                    ),
                )
            )

        return vehicle_matches

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

    @classmethod
    def _try_normalize_bbox(cls, bbox: Any) -> BBox | None:
        """Normalize a bbox, returning None when malformed."""
        try:
            return cls._normalize_bbox(bbox)
        except (TypeError, ValueError):
            return None

    def _draw_slot_status(
        self,
        frame: np.ndarray,
        record: OccupancyRecord,
    ) -> None:
        """Draw one parking slot occupancy record."""
        bbox = self._try_normalize_bbox(record.get("bbox"))
        if bbox is None:
            return

        x1, y1, x2, y2 = bbox
        occupied = bool(record.get("occupied", False))
        color = (
            self.config.occupied_color
            if occupied
            else self.config.free_color
        )
        label_lines = self._slot_label_lines(record, occupied)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self.config.rectangle_thickness,
        )
        self._draw_text_lines(
            frame=frame,
            lines=label_lines,
            origin=(x1 + 4, max(20, y1 - 8)),
            color=color,
            font_scale=self.config.font_scale,
        )

    def _draw_summary(
        self,
        frame: np.ndarray,
        occupancy_data: list[OccupancyRecord],
    ) -> None:
        """Draw available, occupied, and occupancy percentage summary."""
        statistics = self.get_statistics(occupancy_data)

        summary_lines = [
            f"Available: {statistics['available_slots']}",
            f"Occupied: {statistics['occupied_slots']}",
            f"Occupancy: {statistics['occupancy_percentage']:.1f}%",
        ]

        self._draw_text_block(frame, summary_lines, origin=(15, 30))

    def _slot_label_lines(
        self,
        record: OccupancyRecord,
        occupied: bool,
    ) -> list[str]:
        """Build compact per-slot label lines."""
        slot_id = record.get("id", "?")
        label_lines = [f"Slot {slot_id}"]

        if not occupied:
            label_lines.append("Free")
            return label_lines

        vehicle_class = record.get("vehicle_class") or "Vehicle"
        overlap_score = self._optional_float(record.get("overlap_score")) or 0.0
        label_lines.extend(
            [
                str(vehicle_class).title(),
                f"{overlap_score * 100:.0f}%",
            ]
        )
        return label_lines

    def _draw_text_lines(
        self,
        frame: np.ndarray,
        lines: list[str],
        origin: tuple[int, int],
        color: tuple[int, int, int],
        font_scale: float,
    ) -> None:
        """Draw multiple text lines using consistent spacing."""
        x, y = origin
        line_height = max(18, int(26 * font_scale))

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + index * line_height),
                self.FONT,
                font_scale,
                color,
                self.config.text_thickness,
                cv2.LINE_AA,
            )

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
            self._draw_text_lines(
                frame=frame,
                lines=[line],
                origin=(x, y + index * line_height),
                color=self.config.text_color,
                font_scale=self.config.summary_font_scale,
            )

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        """Return a clean string value or None."""
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        """Return a float value or None when conversion fails."""
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_iterable(items: Any) -> list[Any] | Any:
        """Return items when iterable, otherwise an empty list."""
        if items is None:
            return []

        try:
            iter(items)
        except TypeError:
            return []

        return items

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Validate that frame is a non-empty OpenCV-compatible image."""
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array.")

        if frame.size == 0:
            raise ValueError("frame must not be empty.")

        if frame.ndim not in (2, 3):
            raise ValueError("frame must be a 2D or 3D image array.")
