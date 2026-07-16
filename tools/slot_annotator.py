"""Manual parking slot annotation tool for fixed CCTV cameras."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationConfig:
    """Configuration for the parking slot annotator."""

    video_path: Path = Path("data/videos/parking.mp4")
    output_path: Path = Path("data/slots/parking_slots.json")
    window_name: str = "Parking Slot Annotator"
    min_width: int = 20
    min_height: int = 20
    rectangle_color: tuple[int, int, int] = (0, 255, 0)
    preview_color: tuple[int, int, int] = (0, 220, 255)
    text_color: tuple[int, int, int] = (255, 255, 255)
    panel_color: tuple[int, int, int] = (0, 0, 0)
    rectangle_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2


@dataclass(frozen=True)
class ParkingSlot:
    """A manually annotated parking slot rectangle."""

    slot_id: int
    bbox: tuple[int, int, int, int]

    def to_json(self) -> dict[str, int | list[int]]:
        """Convert the slot into the required JSON-serializable format."""
        return {
            "id": self.slot_id,
            "bbox": list(self.bbox),
        }


class ParkingSlotAnnotator:
    """Interactive OpenCV tool for annotating parking slot rectangles."""

    def __init__(self, config: AnnotationConfig | None = None) -> None:
        """
        Initialize the annotator.

        Args:
            config: Optional annotation configuration.
        """
        self.config = config or AnnotationConfig()
        self.frame: np.ndarray | None = None
        self.slots: list[ParkingSlot] = []
        self.is_drawing = False
        self.start_point: tuple[int, int] | None = None
        self.current_point: tuple[int, int] | None = None
        self.saved = False

    def load_first_frame(self) -> np.ndarray:
        """
        Load only the first valid frame from the configured video.

        Returns:
            First valid video frame.

        Raises:
            FileNotFoundError: If the video file is missing.
            RuntimeError: If no valid frame can be read.
        """
        if not self.config.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.config.video_path}"
            )

        capture = cv2.VideoCapture(str(self.config.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open video: {self.config.video_path}"
            )

        try:
            success, frame = capture.read()
        finally:
            capture.release()

        if not success or frame is None or frame.size == 0:
            raise RuntimeError(
                f"Unable to read a valid frame from: {self.config.video_path}"
            )

        self.frame = frame
        return frame

    def mouse_callback(
        self,
        event: int,
        x_coordinate: int,
        y_coordinate: int,
        _flags: int,
        _param: object,
    ) -> None:
        """
        Handle OpenCV mouse events for drawing rectangles.

        Args:
            event: OpenCV mouse event ID.
            x_coordinate: Mouse x-coordinate.
            y_coordinate: Mouse y-coordinate.
            _flags: Unused OpenCV event flags.
            _param: Unused callback parameter.
        """
        point = (x_coordinate, y_coordinate)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_point = point
            self.current_point = point
            return

        if event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.current_point = point
            return

        if event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            self.current_point = point
            self._add_slot_from_points(self.start_point, self.current_point)
            self.start_point = None
            self.current_point = None

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the full annotation overlay.

        Args:
            frame: Base frame to annotate.

        Returns:
            Display frame with slots, preview, and instructions.
        """
        display_frame = frame.copy()
        self.draw_existing_slots(display_frame)
        self._draw_current_rectangle(display_frame)
        self._draw_instruction_panel(display_frame)
        return display_frame

    def draw_existing_slots(self, frame: np.ndarray) -> None:
        """
        Draw all saved slot rectangles and IDs.

        Args:
            frame: Frame to draw on.
        """
        for slot in self.slots:
            x1, y1, x2, y2 = slot.bbox
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self.config.rectangle_color,
                self.config.rectangle_thickness,
            )
            self._draw_text(
                frame,
                f"Slot {slot.slot_id}",
                (x1 + 5, max(22, y1 - 8)),
                self.config.rectangle_color,
            )

    def save_slots(self) -> bool:
        """
        Save annotated slots to JSON.

        Returns:
            True when saving succeeds, otherwise False.
        """
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [slot.to_json() for slot in self.slots]

        try:
            with self.config.output_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=4)
        except OSError as exc:
            print(f"Failed to save parking slots: {exc}")
            return False

        self.saved = True
        print(f"Saved {len(self.slots)} parking slots")
        return True

    def run(self) -> None:
        """Run the interactive annotation GUI."""
        try:
            frame = self.load_first_frame()
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            return

        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.window_name, self.mouse_callback)

        try:
            while True:
                display_frame = self.draw_overlay(frame)
                cv2.imshow(self.config.window_name, display_frame)

                key = cv2.waitKey(20) & 0xFF
                if self._handle_keypress(key):
                    break
        finally:
            cv2.destroyAllWindows()

    def _handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard controls.

        Args:
            key: OpenCV waitKey result.

        Returns:
            True when the application should exit.
        """
        if key in (ord("s"), ord("S")):
            self.save_slots()
            return False

        if key in (ord("r"), ord("R")):
            self._remove_last_slot()
            return False

        if key in (ord("c"), ord("C")):
            self._clear_slots()
            return False

        if key in (ord("q"), ord("Q"), 27):
            return True

        return False

    def _add_slot_from_points(
        self,
        start_point: tuple[int, int] | None,
        end_point: tuple[int, int] | None,
    ) -> None:
        """Validate and store a slot rectangle from two points."""
        if start_point is None or end_point is None:
            return

        bbox = self._normalize_bbox(start_point, end_point)
        if not self._is_valid_bbox(bbox):
            return

        self.slots.append(
            ParkingSlot(
                slot_id=len(self.slots) + 1,
                bbox=bbox,
            )
        )

    def _remove_last_slot(self) -> None:
        """Remove the most recently annotated slot."""
        if self.slots:
            self.slots.pop()

    def _clear_slots(self) -> None:
        """Remove all annotated slots."""
        self.slots.clear()

    def _draw_current_rectangle(self, frame: np.ndarray) -> None:
        """Draw the active rectangle preview while dragging."""
        if (
            not self.is_drawing
            or self.start_point is None
            or self.current_point is None
        ):
            return

        x1, y1, x2, y2 = self._normalize_bbox(
            self.start_point,
            self.current_point,
        )
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.config.preview_color,
            self.config.rectangle_thickness,
        )

    def _draw_instruction_panel(self, frame: np.ndarray) -> None:
        """Draw slot count and keyboard/mouse instructions."""
        instructions = [
            f"Total Slots: {len(self.slots)}",
            "LMB : Draw",
            "S : Save",
            "R : Undo",
            "C : Clear",
            "Q : Quit",
        ]

        x_origin = 12
        y_origin = 28
        line_height = 26
        panel_width = 190
        panel_height = line_height * len(instructions) + 16

        cv2.rectangle(
            frame,
            (8, 8),
            (8 + panel_width, 8 + panel_height),
            self.config.panel_color,
            thickness=-1,
        )

        for index, text in enumerate(instructions):
            self._draw_text(
                frame,
                text,
                (x_origin, y_origin + index * line_height),
                self.config.text_color,
            )

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """Draw text with the configured font."""
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color,
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _normalize_bbox(
        first_point: tuple[int, int],
        second_point: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Normalize two points into x1 < x2 and y1 < y2."""
        x1, y1 = first_point
        x2, y2 = second_point
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _is_valid_bbox(self, bbox: tuple[int, int, int, int]) -> bool:
        """Return True when a rectangle meets minimum size requirements."""
        x1, y1, x2, y2 = bbox
        return (
            x2 - x1 >= self.config.min_width
            and y2 - y1 >= self.config.min_height
        )


if __name__ == "__main__":
    annotator = ParkingSlotAnnotator()
    annotator.run()
