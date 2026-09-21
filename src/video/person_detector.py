"""
YOLO Person Detection Module.
Modular detector supporting YOLOv8 and YOLO11 via Ultralytics.
Specifically extracts person detections (COCO Class ID 0), bounding coordinates,
confidences, counts, and ground contact points for density modeling.
"""

from dataclasses import dataclass
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from src.utils.logger import setup_logger

logger = setup_logger("person_detector")


@dataclass
class DetectionResult:
    """
    Standardized detection output container.
    """
    boxes: np.ndarray             # (N, 4) array in [x1, y1, x2, y2] format
    confidences: np.ndarray       # (N,) array of detection confidence scores [0.0 - 1.0]
    class_ids: np.ndarray         # (N,) array of class IDs (0 for person)
    person_count: int             # Total count of detected persons
    inference_ms: float           # Inference time in milliseconds
    centers: np.ndarray           # (N, 2) array of bbox centers (cx, cy)
    bottom_centers: np.ndarray    # (N, 2) array of ground contact points (cx, y2)


class YOLOPersonDetector:
    """
    Ultralytics YOLO wrapper tailored for high-accuracy person detection in crowded scenes.
    Supports YOLOv8n, YOLOv8s, YOLO11n, and custom trained surveillance weights.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "auto",
        target_classes: Optional[List[int]] = None,
        imgsz: int = 640,
    ):
        self.model_path = model_path
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.target_classes = target_classes if target_classes is not None else [0]  # 0 is 'person' in COCO
        self.imgsz = imgsz

        self.device = self._select_device(device)
        self.model = self._load_model()

    def _select_device(self, requested_device: str) -> str:
        """Determines computation device with fallback to CPU if CUDA is unavailable."""
        req = requested_device.lower().strip()
        if req in ("cuda", "gpu") or req.startswith("cuda:"):
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {gpu_name}")
                return "cuda:0" if req in ("cuda", "gpu") else req
            else:
                logger.warning("CUDA was requested but is not available on this machine. Falling back to CPU.")
                return "cpu"

        if req == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Auto-selected CUDA device: {gpu_name}")
                return "cuda:0"
            else:
                logger.info("Auto-selected CPU execution (CUDA not available).")
                return "cpu"

        return "cpu"

    def _load_model(self):
        """Loads the YOLO model via Ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please run: pip install ultralytics"
            ) from e

        logger.info(f"Loading YOLO model weights: '{self.model_path}' on device: '{self.device}'...")
        try:
            # Model path can be a local path or auto-downloadable tag ('yolov8n.pt', 'yolo11n.pt')
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info(f"Successfully loaded {self.model_path} ({model.names[0] if hasattr(model, 'names') else 'COCO'}).")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            raise RuntimeError(
                f"Could not load YOLO model '{self.model_path}'. "
                "Ensure internet access is available to download pre-trained weights or provide a valid local .pt file."
            ) from e

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Executes person detection on an input BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3).

        Returns:
            DetectionResult dataclass containing bounding boxes, confidences, counts, and timings.
        """
        start_time = time.perf_counter()

        # Run inference with Ultralytics
        # classes=[0] filters COCO person class directly in the model head
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - start_time) * 1000.0

        if not results or len(results) == 0:
            return self._empty_result(inference_ms)

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return self._empty_result(inference_ms)

        # Extract coordinates [x1, y1, x2, y2]
        boxes_xyxy = boxes_obj.xyxy.cpu().numpy()
        confidences = boxes_obj.conf.cpu().numpy()
        class_ids = boxes_obj.cls.cpu().numpy().astype(int)

        # Compute center points and ground contact (bottom-center) points
        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
        bottom_y = boxes_xyxy[:, 3]

        centers = np.column_stack((cx, cy))
        bottom_centers = np.column_stack((cx, bottom_y))

        return DetectionResult(
            boxes=boxes_xyxy,
            confidences=confidences,
            class_ids=class_ids,
            person_count=len(boxes_xyxy),
            inference_ms=inference_ms,
            centers=centers,
            bottom_centers=bottom_centers,
        )

    def _empty_result(self, inference_ms: float) -> DetectionResult:
        """Returns an empty result when no persons are detected."""
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            confidences=np.empty((0,), dtype=np.float32),
            class_ids=np.empty((0,), dtype=int),
            person_count=0,
            inference_ms=inference_ms,
            centers=np.empty((0, 2), dtype=np.float32),
            bottom_centers=np.empty((0, 2), dtype=np.float32),
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        box_color: Tuple[int, int, int] = (0, 220, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Draws bounding boxes, labels, and bottom-center anchor points onto the frame.
        """
        annotated = frame.copy()

        for i in range(detection.person_count):
            x1, y1, x2, y2 = detection.boxes[i].astype(int)
            conf = detection.confidences[i]
            bx, by = detection.bottom_centers[i].astype(int)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Draw ground foot anchor dot
            cv2.circle(annotated, (bx, by), 4, (0, 0, 255), -1)

            # Draw label banner
            if show_conf:
                label = f"Person {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_ymin = max(y1 - lh - 4, 0)

                cv2.rectangle(annotated, (x1, label_ymin), (x1 + lw + 4, label_ymin + lh + 4), box_color, -1)
                cv2.putText(annotated, label, (x1 + 2, label_ymin + lh + 1), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return annotated
