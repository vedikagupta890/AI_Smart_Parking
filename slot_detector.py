"""Parking slot detection architecture for the AI Smart Parking System."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SlotDetectionConfig:
    """Configuration values for the future parking slot detection pipeline."""

    blur_kernel_size: tuple[int, int] = (5, 5)
    blur_sigma: float = 0.0
    canny_threshold_low: int = 50
    canny_threshold_high: int = 150
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10


class ParkingSlotDetector:
    """
    Architecture for detecting parking slot candidates from video frames.

    The class intentionally defines the full processing pipeline without
    implementing the underlying computer vision algorithms yet.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the parking slot detector.

        Args:
            debug: Whether future implementations should store intermediate
                images and additional diagnostic data.
        """
        self.debug = debug
        self.config = SlotDetectionConfig()
        self.frame: np.ndarray | None = None
        self.debug_images: dict[str, np.ndarray] = {}
        self._opencv_module = cv2

    def load_frame(self, frame: np.ndarray) -> None:
        """
        Validate and store an OpenCV frame for slot detection.

        Args:
            frame: Input OpenCV image frame in BGR format.

        Raises:
            NotImplementedError: Until frame validation and storage are added.
        """
        raise NotImplementedError(
            "Frame validation and storage are not implemented yet."
        )

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a frame for edge detection.

        Future implementation will convert the frame to grayscale, apply
        Gaussian blur, and return the processed image.

        Args:
            frame: Input OpenCV image frame in BGR format.

        Returns:
            Processed grayscale image.

        Raises:
            NotImplementedError: Until preprocessing logic is added.
        """
        raise NotImplementedError(
            "Grayscale conversion and Gaussian blur are not implemented yet."
        )

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in a preprocessed image using Canny edge detection.

        Args:
            image: Preprocessed grayscale image.

        Returns:
            Binary edge image.

        Raises:
            NotImplementedError: Until Canny edge detection is added.
        """
        raise NotImplementedError(
            "Canny edge detection is not implemented yet."
        )

    def detect_lines(self, edge_image: np.ndarray) -> Any:
        """
        Detect parking line candidates using Probabilistic Hough Transform.

        Args:
            edge_image: Binary edge image from detect_edges().

        Returns:
            Detected parking line candidates.

        Raises:
            NotImplementedError: Until Hough line detection is added.
        """
        raise NotImplementedError(
            "Probabilistic Hough line detection is not implemented yet."
        )

    def extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract the parking area from a frame.

        Future implementation will support configurable custom regions of
        interest. The initial algorithmic implementation may return the
        original frame.

        Args:
            frame: Input OpenCV image frame in BGR format.

        Returns:
            Region of interest image.

        Raises:
            NotImplementedError: Until ROI extraction is added.
        """
        raise NotImplementedError("ROI extraction is not implemented yet.")

    def detect_slots(self, frame: np.ndarray) -> list[Any]:
        """
        Run the full parking slot detection pipeline.

        Future implementation will call load_frame(), extract_roi(),
        preprocess(), detect_edges(), and detect_lines() to produce parking
        slot candidates.

        Args:
            frame: Input OpenCV image frame in BGR format.

        Returns:
            Detected parking slot candidates.

        Raises:
            NotImplementedError: Until the slot detection pipeline is added.
        """
        raise NotImplementedError(
            "Parking slot detection pipeline is not implemented yet."
        )

    def draw_slots(self, frame: np.ndarray, slots: list[Any]) -> np.ndarray:
        """
        Draw detected parking slots on a frame.

        Args:
            frame: Input OpenCV image frame in BGR format.
            slots: Parking slot candidates returned by detect_slots().

        Returns:
            Annotated frame.

        Raises:
            NotImplementedError: Until slot drawing logic is added.
        """
        raise NotImplementedError("Parking slot drawing is not implemented yet.")

    def save_debug_images(self) -> None:
        """
        Save intermediate debug images produced by the pipeline.

        Raises:
            NotImplementedError: Until debug image export is added.
        """
        raise NotImplementedError(
            "Saving debug images is not implemented yet."
        )
