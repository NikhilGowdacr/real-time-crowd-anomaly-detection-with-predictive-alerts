"""
Video Input Capture Module.
Supports local video files, webcams, RTSP / network IP streams, and synthetic video generation.
"""

import os
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("video_capture")


class SyntheticCrowdGenerator:
    """
    Generates synthetic video frames containing moving crowd agents.
    Useful for automated unit testing, CI pipelines, and environments without CCTV cameras or webcams.
    """

    def __init__(self, width: int = 1280, height: int = 720, num_people: int = 18):
        self.width = width
        self.height = height
        self.num_people = num_people
        self.frame_idx = 0

        # Initialize simulated people with positions, velocities, and sizes
        np.random.seed(42)
        self.people = []
        for i in range(num_people):
            x = np.random.uniform(100, width - 100)
            y = np.random.uniform(150, height - 150)
            vx = np.random.uniform(-2.5, 2.5)
            vy = np.random.uniform(-1.5, 1.5)
            # Size roughly scales with perspective (lower = closer = larger)
            scale = 0.6 + 0.6 * (y / height)
            w = int(45 * scale)
            h = int(120 * scale)
            color = (
                int(np.random.randint(60, 220)),
                int(np.random.randint(60, 220)),
                int(np.random.randint(60, 220)),
            )
            self.people.append({"x": x, "y": y, "vx": vx, "vy": vy, "w": w, "h": h, "color": color})

    def generate_frame(self) -> np.ndarray:
        """Draws a synthetic surveillance frame with moving human figures."""
        # Surveillance background: courtyard floor with tiles
        frame = np.full((self.height, self.width, 3), 215, dtype=np.uint8)

        # Draw grid tile lines
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (200, 200, 200), 1)
        for y in range(0, self.height, 60):
            cv2.line(frame, (0, y), (self.width, y), (200, 200, 200), 1)

        # Update and draw simulated people (head, torso, legs)
        for p in self.people:
            # Update position
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Boundary bounce
            if p["x"] < 50 or p["x"] > self.width - 50:
                p["vx"] *= -1
            if p["y"] < 100 or p["y"] > self.height - 100:
                p["vy"] *= -1

            # Recalculate scale based on perspective
            scale = 0.6 + 0.6 * (p["y"] / self.height)
            w = int(45 * scale)
            h = int(120 * scale)

            cx = int(p["x"])
            cy = int(p["y"])

            # Shadow
            cv2.ellipse(frame, (cx, cy + h // 2), (w // 2, 8), 0, 0, 360, (170, 170, 170), -1)

            # Body (torso)
            torso_top = cy - h // 4
            torso_bottom = cy + h // 4
            cv2.rectangle(frame, (cx - w // 3, torso_top), (cx + w // 3, torso_bottom), p["color"], -1)

            # Legs
            cv2.line(frame, (cx - w // 5, torso_bottom), (cx - w // 5, cy + h // 2), (40, 40, 40), max(2, int(4 * scale)))
            cv2.line(frame, (cx + w // 5, torso_bottom), (cx + w // 5, cy + h // 2), (40, 40, 40), max(2, int(4 * scale)))

            # Head
            head_radius = max(6, int(14 * scale))
            cv2.circle(frame, (cx, torso_top - head_radius), head_radius, (190, 210, 235), -1)
            cv2.circle(frame, (cx, torso_top - head_radius), head_radius, (30, 30, 30), 1)

        self.frame_idx += 1
        return frame


class VideoSource:
    """
    Robust video capture supporting Video files, Webcams, RTSP streams, and Synthetic mode.
    """

    def __init__(
        self,
        source: Union[str, int] = "synthetic",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
        loop: bool = True,
    ):
        """
        Initialize video capture.

        Args:
            source: Video file path, integer webcam index (e.g. 0), RTSP url, or "synthetic".
            width: Desired capture frame width (optional).
            height: Desired capture frame height (optional).
            fps: Desired capture frames per second.
            loop: If True and source is a file, loops back to the start when reaching EOF.
        """
        self.raw_source = source
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps
        self.loop = loop

        self.is_synthetic = False
        self.synthetic_gen: Optional[SyntheticCrowdGenerator] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_file = False
        self.is_stream = False

        self._initialize_source()

    def _initialize_source(self) -> None:
        """Parses and opens the requested video input source."""
        str_source = str(self.raw_source).strip()

        # 1. Synthetic generator mode
        if str_source.lower() in ("synthetic", "demo", "simulated", "mock"):
            logger.info("Initializing Synthetic Video Generator (18 simulated agents)...")
            self.is_synthetic = True
            w = self.desired_width or 1280
            h = self.desired_height or 720
            self.synthetic_gen = SyntheticCrowdGenerator(width=w, height=h, num_people=18)
            return

        # 2. Integer Webcam device index
        if str_source.isdigit() or isinstance(self.raw_source, int):
            device_id = int(str_source)
            logger.info(f"Connecting to Webcam device index: {device_id}...")
            # On Windows, cv2.CAP_DSHOW provides fast, reliable camera opening
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(device_id)

            if not self.cap.isOpened():
                raise ConnectionError(
                    f"Unable to access webcam device index {device_id}. "
                    "Ensure the camera is plugged in, not used by another application, "
                    "or try source: 'synthetic'."
                )
            self._apply_camera_settings()
            return

        # 3. RTSP / HTTP Camera stream
        if str_source.startswith(("rtsp://", "http://", "https://")):
            logger.info(f"Connecting to Network/RTSP stream: {str_source}...")
            self.is_stream = True
            self.cap = cv2.VideoCapture(str_source)
            if not self.cap.isOpened():
                raise ConnectionError(
                    f"Failed to connect to RTSP/network video stream: '{str_source}'. "
                    "Check network connectivity, credentials, and RTSP stream URL."
                )
            return

        # 4. Local video file
        file_path = Path(str_source)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Video file not found at path: '{file_path.resolve()}'. "
                f"Please place your video file in data/videos/ or specify source: 'synthetic' or 0 (webcam)."
            )

        logger.info(f"Opening video file: {file_path.resolve()}...")
        self.is_file = True
        self.cap = cv2.VideoCapture(str(file_path.resolve()))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file '{file_path}'. Codec may be unsupported.")

    def _apply_camera_settings(self) -> None:
        """Sets hardware properties if requested."""
        if self.cap is not None and self.cap.isOpened():
            if self.desired_width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
            if self.desired_height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
            if self.desired_fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame from the source.

        Returns:
            Tuple of (success, frame). If EOF and looping, automatically rewinds.
        """
        if self.is_synthetic:
            assert self.synthetic_gen is not None
            frame = self.synthetic_gen.generate_frame()
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.is_file and self.loop:
                logger.info("End of video reached. Looping back to beginning...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                return ret, frame
            return False, None

        return True, frame

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Allows iterating directly over the capture instance."""
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    @property
    def fps(self) -> float:
        """Return FPS of video source."""
        if self.is_synthetic:
            return float(self.desired_fps)
        if self.cap and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            return val if val > 0 else float(self.desired_fps)
        return float(self.desired_fps)

    @property
    def frame_count(self) -> int:
        """Return total frame count if source is a file, else -1."""
        if self.cap and self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    @property
    def width(self) -> int:
        if self.is_synthetic and self.synthetic_gen:
            return self.synthetic_gen.width
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self.desired_width or 1280

    @property
    def height(self) -> int:
        if self.is_synthetic and self.synthetic_gen:
            return self.synthetic_gen.height
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.desired_height or 720

    @property
    def is_opened(self) -> bool:
        if self.is_synthetic:
            return True
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Releases capture resources cleanly."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video capture source successfully released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
