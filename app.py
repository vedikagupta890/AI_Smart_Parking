"""Flask application for the AI Smart Parking System."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

from flask import Flask, Response, jsonify, render_template
from werkzeug.exceptions import BadRequest, InternalServerError, NotFound

from parking import SmartParkingPipeline, VIDEO_PATH

HOST = "0.0.0.0"
PORT = 5000
DEBUG = False
STREAM_MIMETYPE = "multipart/x-mixed-replace; boundary=frame"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline: SmartParkingPipeline | None = None


def create_pipeline() -> SmartParkingPipeline:
    """
    Create the shared SmartParkingPipeline singleton.

    Returns:
        Initialized SmartParkingPipeline instance.

    Raises:
        RuntimeError: If the pipeline cannot be initialized.
    """
    try:
        smart_parking_pipeline = SmartParkingPipeline()
    except Exception as exc:
        raise RuntimeError(f"Pipeline initialization failed: {exc}") from exc

    logger.info("Pipeline initialized")
    return smart_parking_pipeline


def initialize_pipeline() -> None:
    """Initialize the global pipeline once at application startup."""
    global pipeline

    if pipeline is not None:
        return

    try:
        pipeline = create_pipeline()
    except RuntimeError as exc:
        pipeline = None
        logger.error("%s", exc)


def validate_video_source(video_path: str) -> Path:
    """
    Validate the configured video source path.

    Args:
        video_path: Path to the input video source.

    Returns:
        Path object for the validated source.

    Raises:
        FileNotFoundError: If the video source does not exist.
    """
    source_path = Path(video_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Video source not found: {video_path}")

    return source_path


def generate_frames() -> Generator[bytes, None, None]:
    """
    Delegate MJPEG frame generation to SmartParkingPipeline.

    Yields:
        Multipart MJPEG frame bytes produced by the pipeline.

    Raises:
        RuntimeError: If the pipeline is unavailable.
        FileNotFoundError: If the configured video source is missing.
    """
    if pipeline is None:
        raise RuntimeError("Pipeline is not initialized.")

    validate_video_source(VIDEO_PATH)
    logger.info("Client connected to video stream")

    try:
        yield from pipeline.generate_frames(VIDEO_PATH)
    finally:
        logger.info("Client disconnected")


def get_current_statistics() -> dict[str, int | float]:
    """
    Return the latest statistics from the running pipeline.

    Returns:
        The pipeline's latest_statistics dictionary.
    """
    if pipeline is None:
        return {}

    return pipeline.latest_statistics.copy()


def json_error(message: str, status_code: int) -> tuple[Response, int]:
    """
    Build a JSON error response.

    Args:
        message: Human-readable error message.
        status_code: HTTP status code.

    Returns:
        Flask JSON response tuple.
    """
    return jsonify({"error": message}), status_code


@app.route("/")
def index() -> str | tuple[Response, int]:
    """
    Render the dashboard page.

    Returns:
        Rendered templates/index.html, or JSON error response.
    """
    try:
        return render_template("index.html")
    except Exception as exc:
        logger.error("Failed to render index.html: %s", exc)
        return json_error("Unable to render dashboard.", 500)


@app.route("/video_feed")
def video_feed() -> Response | tuple[Response, int]:
    """
    Return the live multipart MJPEG stream.

    Returns:
        Streaming response, or JSON error response.
    """
    if pipeline is None:
        logger.error("Video feed requested before pipeline initialization.")
        return json_error("Pipeline is not initialized.", 503)

    try:
        validate_video_source(VIDEO_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return json_error(str(exc), 404)

    try:
        return Response(generate_frames(), mimetype=STREAM_MIMETYPE)
    except RuntimeError as exc:
        logger.error("Unable to start video stream: %s", exc)
        return json_error(str(exc), 503)
    except Exception as exc:
        logger.error("Unexpected video stream error: %s", exc)
        return json_error("Unable to start video stream.", 500)


@app.route("/api/status")
def api_status() -> Response | tuple[Response, int]:
    """
    Return current parking statistics.

    Returns:
        JSON response with pipeline.latest_statistics.
    """
    if pipeline is None:
        logger.warning("Status requested before pipeline initialization.")
        return json_error("Pipeline is not initialized.", 503)

    return jsonify(get_current_statistics())


@app.route("/health")
def health() -> Response:
    """
    Return application health information.

    Returns:
        JSON response containing app and pipeline health.
    """
    return jsonify(
        {
            "status": "ok",
            "pipeline_initialized": pipeline is not None,
            "video_source": VIDEO_PATH,
        }
    )


@app.errorhandler(BadRequest)
def handle_bad_request(error: BadRequest) -> tuple[Response, int]:
    """Return JSON for HTTP 400 errors."""
    logger.warning("Bad request: %s", error)
    return json_error("Bad request.", 400)


@app.errorhandler(NotFound)
def handle_not_found(error: NotFound) -> tuple[Response, int]:
    """Return JSON for HTTP 404 errors."""
    logger.warning("Not found: %s", error)
    return json_error("Resource not found.", 404)


@app.errorhandler(InternalServerError)
def handle_internal_server_error(
    error: InternalServerError,
) -> tuple[Response, int]:
    """Return JSON for HTTP 500 errors."""
    logger.error("Internal server error: %s", error)
    return json_error("Internal server error.", 500)


initialize_pipeline()


if __name__ == "__main__":
    logger.info("Server started")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
