"""Utilities for loading and validating parking slot configuration files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


DEFAULT_SLOT_PATH = Path("data/slots/parking_slots.json")
MIN_SLOT_WIDTH = 20
MIN_SLOT_HEIGHT = 20


@dataclass(frozen=True)
class ParkingSlot:
    """Parking slot definition loaded from JSON configuration."""

    id: int
    name: str
    bbox: tuple[int, int, int, int]


def load_slots(json_path: str | Path = DEFAULT_SLOT_PATH) -> list[ParkingSlot]:
    """
    Load parking slots from a JSON file.

    Args:
        json_path: Path to the parking slot JSON file.

    Returns:
        Validated list of parking slots.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON content or slot data is invalid.
        OSError: If the file cannot be read.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Parking slot JSON not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid parking slot JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read parking slot JSON {path}: {exc}") from exc

    slots = _parse_slots(raw_data)
    validate_slots(slots)
    return slots


def save_slots(
    slots: list[ParkingSlot],
    json_path: str | Path = DEFAULT_SLOT_PATH,
) -> None:
    """
    Validate and save parking slots to a formatted JSON file.

    Args:
        slots: Parking slots to save.
        json_path: Destination JSON file path.

    Raises:
        ValueError: If the slot list is invalid.
        OSError: If the JSON file cannot be written.
    """
    validate_slots(slots)

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_slot_to_json(slot) for slot in slots]

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
    except OSError as exc:
        raise OSError(f"Unable to write parking slot JSON {path}: {exc}") from exc


def validate_slots(slots: list[ParkingSlot]) -> None:
    """
    Validate parking slot definitions.

    Checks duplicate IDs, duplicate names, bbox structure, coordinate order,
    and minimum rectangle dimensions.

    Args:
        slots: Parking slots to validate.

    Raises:
        ValueError: If the slot list or any slot is invalid.
        TypeError: If the input is not a list of ParkingSlot objects.
    """
    if not isinstance(slots, list):
        raise TypeError("Parking slots must be provided as a list")

    if not slots:
        raise ValueError("Parking slot JSON is empty")

    seen_ids: set[int] = set()
    seen_names: set[str] = set()

    for index, slot in enumerate(slots, start=1):
        if not isinstance(slot, ParkingSlot):
            raise TypeError(
                f"Slot at position {index} must be a ParkingSlot instance"
            )

        _validate_slot_identity(slot, index, seen_ids, seen_names)
        _validate_bbox(slot.bbox, slot.id)


def slots_to_bboxes(
    slots: list[ParkingSlot],
) -> list[tuple[int, int, int, int]]:
    """
    Convert parking slots to bbox tuples for occupancy detection.

    Args:
        slots: Parking slots to convert.

    Returns:
        List of bbox tuples.
    """
    validate_slots(slots)
    return [slot.bbox for slot in slots]


def slot_count(slots: list[ParkingSlot]) -> int:
    """
    Return the number of parking slots.

    Args:
        slots: Parking slots to count.

    Returns:
        Total number of parking slots.
    """
    validate_slots(slots)
    return len(slots)


def _parse_slots(raw_data: Any) -> list[ParkingSlot]:
    """Parse raw JSON data into ParkingSlot objects."""
    if not isinstance(raw_data, list):
        raise ValueError("Parking slot JSON must contain a list of slots")

    if not raw_data:
        raise ValueError("Parking slot JSON is empty")

    return [
        _parse_slot(slot_data, index)
        for index, slot_data in enumerate(raw_data, start=1)
    ]


def _parse_slot(slot_data: Any, index: int) -> ParkingSlot:
    """Parse and normalize one raw slot dictionary."""
    if not isinstance(slot_data, dict):
        raise ValueError(f"Slot at position {index} must be a JSON object")

    try:
        slot_id = int(slot_data["id"])
        slot_name = slot_data.get(
            "name",
            f"A{slot_id}",
        )
        bbox = slot_data["bbox"]
        
    except KeyError as exc:
        raise ValueError(
            f"Slot at position {index} is missing required field: {exc.args[0]}"
        ) from exc

    if not _is_plain_integer(slot_id):
        raise ValueError(f"Slot at position {index} has invalid id: {slot_id!r}")

    if not isinstance(slot_name, str) or not slot_name.strip():
        raise ValueError(
            f"Slot {slot_id} has invalid name: expected a non-empty string"
        )

    return ParkingSlot(
        id=slot_id,
        name=slot_name.strip(),
        bbox=_parse_bbox(bbox, slot_id),
    )


def _parse_bbox(raw_bbox: Any, slot_id: int) -> tuple[int, int, int, int]:
    """Parse one raw bbox value into a coordinate tuple."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError(
            f"Slot {slot_id} bbox must contain exactly four integers"
        )

    if not all(_is_plain_integer(value) for value in raw_bbox):
        raise ValueError(f"Slot {slot_id} bbox must contain only integers")

    return tuple(raw_bbox)


def _validate_slot_identity(
    slot: ParkingSlot,
    index: int,
    seen_ids: set[int],
    seen_names: set[str],
) -> None:
    """Validate slot ID/name and track duplicates."""
    if not _is_plain_integer(slot.id):
        raise ValueError(f"Slot at position {index} has invalid id: {slot.id!r}")

    if slot.id in seen_ids:
        raise ValueError(f"Duplicate parking slot id found: {slot.id}")
    seen_ids.add(slot.id)

    if not isinstance(slot.name, str) or not slot.name.strip():
        raise ValueError(
            f"Slot {slot.id} has invalid name: expected a non-empty string"
        )

    normalized_name = slot.name.strip()
    if normalized_name in seen_names:
        raise ValueError(f"Duplicate parking slot name found: {normalized_name}")
    seen_names.add(normalized_name)


def _validate_bbox(bbox: tuple[int, int, int, int], slot_id: int) -> None:
    """Validate bbox shape, coordinate order, and dimensions."""
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise ValueError(
            f"Slot {slot_id} bbox must be a tuple of exactly four integers"
        )

    if not all(_is_plain_integer(value) for value in bbox):
        raise ValueError(f"Slot {slot_id} bbox must contain only integers")

    x1, y1, x2, y2 = bbox

    if x1 >= x2:
        raise ValueError(f"Slot {slot_id} bbox must satisfy x1 < x2")

    if y1 >= y2:
        raise ValueError(f"Slot {slot_id} bbox must satisfy y1 < y2")

    width = x2 - x1
    height = y2 - y1

    if width <= 0:
        raise ValueError(
            f"Slot {slot_id} width must be greater than 0 pixels"
        )

    if height <= 0:
        raise ValueError(
            f"Slot {slot_id} height must be greater than 0 pixels"
        )


def _slot_to_json(slot: ParkingSlot) -> dict[str, int | str | list[int]]:
    """Convert a ParkingSlot into the required JSON object shape."""
    slot_data = asdict(slot)
    slot_data["bbox"] = list(slot.bbox)
    return slot_data


def _is_plain_integer(value: Any) -> bool:
    """Return True for integers while rejecting booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


if __name__ == "__main__":
    slots = load_slots("data/slots/parking_slots.json")
    print(f"Loaded {slot_count(slots)} parking slots")
