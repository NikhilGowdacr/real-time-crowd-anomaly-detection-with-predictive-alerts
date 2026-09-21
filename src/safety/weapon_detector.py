"""
Weapon and Dangerous Object Detection Module.
Provides modular object detection with support for custom-trained YOLO models
(e.g., firearm, knife, dangerous_object) and spatial association with tracked persons.

Research & Academic Integrity Notice:
Pre-trained COCO YOLO weights do NOT provide reliable weapon detection classes.
This module strictly requires a dedicated custom-trained model or gracefully reports
an unconfigured state.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.utils.logger import setup_logger
from src.video.crowd_tracker import TrackedPerson

logger = setup_logger("weapon_detector")


@dataclass
class DetectedObject:
    """
    Representation of a detected dangerous object.
    """
    object_class: str                          # "firearm", "knife", "dangerous_object"
    confidence: float                          # [0.0 - 1.0]
    bbox: np.ndarray                           # [x1, y1, x2, y2]
    timestamp: float                           # Detection epoch timestamp
    center: Tuple[float, float]                # (cx, cy)
    associated_track_id: Optional[int] = None  # ID of nearby tracked person if correlated
    association_confidence: float = 0.0        # Confidence in track association [0.0 - 1.0]
    association_status: str = "isolated"       # "associated", "uncertain", "isolated"


@dataclass
class WeaponDetectionResult:
    """
    Container for weapon detection outcomes in a single frame.
    """
    is_configured: bool
    status_message: str
    detected_objects: List[DetectedObject] = field(default_factory=list)
    max_confidence: float = 0.0
    weapon_score: float = 0.0                  # Normalized safety risk score [0.0 - 1.0]
    inference_ms: float = 0.0


class WeaponDetector:
    """
    Modular dangerous object detector.
    Supports custom-trained YOLO checkpoints, graceful unconfigured fallback,
    and spatial association with tracked pedestrians.
    """

    DEFAULT_CLASSES = ["firearm", "knife", "dangerous_object"]

    def __init__(
        self,
        model_path: Optional[str] = "models/detection/weapon_model.pt",
        confidence: float = 0.50,
        enabled: bool = True,
        target_classes: Optional[List[str]] = None,
        device: str = "auto",
        simulate_for_demo: bool = False,
    ):
        self.enabled = enabled
        self.confidence = float(confidence)
        self.target_classes = target_classes or self.DEFAULT_CLASSES
        self.device = device
        self.simulate_for_demo = simulate_for_demo
        self.model_path = model_path
        self.is_configured = False
        self.model = None

        if not self.enabled:
            self.status_message = "Weapon detection disabled in configuration."
            logger.info(self.status_message)
            return

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads custom YOLO weapon model or configures graceful unconfigured state."""
        if not self.model_path:
            self.status_message = "No weapon model path specified in configuration."
            logger.warning(f"WeaponDetector: {self.status_message}")
            return

        resolved = Path(self.model_path)
        if not resolved.exists():
            self.is_configured = False
            self.status_message = (
                f"Weapon model checkpoint not found at '{resolved}'. "
                "COCO models cannot detect weapons. Module is in UNCONFIGURED state."
            )
            logger.warning(f"WeaponDetector: {self.status_message}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(resolved))
            # Resolve device
            if self.device != "cpu":
                import torch
                dev = "cuda:0" if torch.cuda.is_available() and self.device in ("auto", "cuda") else "cpu"
            else:
                dev = "cpu"
            self.model.to(dev)
            self.is_configured = True
            self.status_message = f"Custom weapon model successfully loaded from '{resolved}'."
            logger.info(self.status_message)
        except Exception as e:
            self.is_configured = False
            self.status_message = f"Failed to initialize weapon model from '{resolved}': {e}"
            logger.error(f"WeaponDetector: {self.status_message}")

    def detect(
        self,
        frame: np.ndarray,
        tracked_persons: Optional[List[TrackedPerson]] = None,
        timestamp: Optional[float] = None,
        demo_trigger: bool = False,
    ) -> WeaponDetectionResult:
        """
        Executes weapon detection and associates findings with tracked persons.

        Args:
            frame: Input BGR frame.
            tracked_persons: List of currently tracked persons for spatial association.
            timestamp: Epoch timestamp.
            demo_trigger: Flag to trigger simulated detection for verification/demo.

        Returns:
            WeaponDetectionResult dataclass.
        """
        current_time = timestamp if timestamp is not None else time.time()
        start_t = time.perf_counter()

        if not self.enabled:
            return WeaponDetectionResult(
                is_configured=False,
                status_message="Disabled in configuration.",
                detected_objects=[],
                weapon_score=0.0,
            )

        # 1. Unconfigured state with demo/simulation handling
        if not self.is_configured:
            # If demo simulation is explicitly active or demo_trigger is true
            if (self.simulate_for_demo or demo_trigger) and tracked_persons:
                return self._generate_simulated_detection(frame, tracked_persons, current_time)

            return WeaponDetectionResult(
                is_configured=False,
                status_message="Weapon model not configured (trained weights required)",
                detected_objects=[],
                max_confidence=0.0,
                weapon_score=0.0,
                inference_ms=0.0,
            )

        # 2. Inference with loaded custom YOLO model
        detected_objects: List[DetectedObject] = []
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - start_t) * 1000.0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
                names = self.model.names if hasattr(self.model, "names") else {}

                for i in range(len(boxes)):
                    c_id = cls_indices[i]
                    cls_name = names.get(c_id, f"class_{c_id}").lower()

                    # Filter by configured target classes if specified
                    if any(target in cls_name for target in self.target_classes):
                        bbox = boxes[i]
                        cx = float((bbox[0] + bbox[2]) / 2.0)
                        cy = float((bbox[1] + bbox[3]) / 2.0)
                        conf = float(confs[i])

                        obj = DetectedObject(
                            object_class=cls_name.upper(),
                            confidence=conf,
                            bbox=bbox,
                            timestamp=current_time,
                            center=(cx, cy),
                        )
                        detected_objects.append(obj)

        except Exception as e:
            logger.error(f"Inference error in weapon detection: {e}")
            inference_ms = (time.perf_counter() - start_t) * 1000.0

        # 3. Associate detected objects with nearby tracked persons
        if tracked_persons:
            self._associate_with_tracks(detected_objects, tracked_persons)

        max_conf = max([o.confidence for o in detected_objects], default=0.0)
        weapon_score = min(1.0, max_conf)

        return WeaponDetectionResult(
            is_configured=True,
            status_message=f"Detected {len(detected_objects)} dangerous object(s).",
            detected_objects=detected_objects,
            max_confidence=max_conf,
            weapon_score=weapon_score,
            inference_ms=inference_ms,
        )

    def _associate_with_tracks(
        self,
        detected_objects: List[DetectedObject],
        tracked_persons: List[TrackedPerson],
    ) -> None:
        """
        Calculates spatial proximity between detected objects and tracked people.
        Adheres to research ethics: reports visual proximity without criminal labeling.
        """
        for obj in detected_objects:
            ox, oy = obj.center
            best_dist = float("inf")
            best_person = None

            for person in tracked_persons:
                px1, py1, px2, py2 = person.current_bbox
                person_w = px2 - px1
                person_h = py2 - py1

                # Distance from object center to person center
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                dist = math.hypot(ox - pcx, oy - pcy)

                # Check if object is within or immediately adjacent to person bounding box
                margin_x = person_w * 0.8
                margin_y = person_h * 0.4
                in_vicinity = (
                    (px1 - margin_x <= ox <= px2 + margin_x) and
                    (py1 - margin_y <= oy <= py2 + margin_y)
                )

                if in_vicinity and dist < best_dist:
                    best_dist = dist
                    best_person = person

            if best_person is not None:
                # Calculate association confidence based on relative proximity
                px1, _, px2, _ = best_person.current_bbox
                p_width = max(20.0, px2 - px1)
                normalized_dist = best_dist / p_width

                if normalized_dist < 0.9:
                    obj.associated_track_id = best_person.track_id
                    obj.association_confidence = round(max(0.4, 1.0 - (normalized_dist * 0.5)), 2)
                    obj.association_status = "associated"
                else:
                    obj.associated_track_id = best_person.track_id
                    obj.association_confidence = 0.45
                    obj.association_status = "uncertain"
            else:
                obj.associated_track_id = None
                obj.association_confidence = 0.0
                obj.association_status = "isolated"

    def _generate_simulated_detection(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson],
        timestamp: float,
    ) -> WeaponDetectionResult:
        """
        Generates simulated weapon detection for controlled testing and demonstration.
        Attaches a mock firearm near the first active tracked person.
        """
        target = tracked_persons[0]
        x1, y1, x2, y2 = target.current_bbox
        # Place mock object near right hand/waist
        obj_w, obj_h = 35, 25
        ox1 = float(min(frame.shape[1] - obj_w - 5, x2 - 10))
        oy1 = float(y1 + (y2 - y1) * 0.5)
        ox2 = ox1 + obj_w
        oy2 = oy1 + obj_h

        mock_obj = DetectedObject(
            object_class="FIREARM",
            confidence=0.87,
            bbox=np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
            timestamp=timestamp,
            center=((ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0),
            associated_track_id=target.track_id,
            association_confidence=0.85,
            association_status="associated",
        )

        return WeaponDetectionResult(
            is_configured=True,
            status_message="[DEMO SIMULATION] Detected simulated dangerous object.",
            detected_objects=[mock_obj],
            max_confidence=0.87,
            weapon_score=0.87,
            inference_ms=1.5,
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        result: WeaponDetectionResult,
        color: Tuple[int, int, int] = (0, 70, 255),  # Deep Orange/Red
    ) -> np.ndarray:
        """
        Draws bounding boxes, object labels, and track association indicators.
        """
        if not result.detected_objects:
            return frame

        annotated = frame.copy()
        for obj in result.detected_objects:
            x1, y1, x2, y2 = obj.bbox.astype(int)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Header text: e.g. FIREARM 87%
            header_text = f"{obj.object_class} {int(obj.confidence * 100)}%"

            # Association subtitle
            if obj.association_status == "associated":
                assoc_text = f"Near Track ID #{obj.associated_track_id}"
            elif obj.association_status == "uncertain":
                assoc_text = "Object detected - person association uncertain"
            else:
                assoc_text = "Isolated object"

            # Draw label box
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw1, th1), _ = cv2.getTextSize(header_text, font, 0.50, 2)
            (tw2, th2), _ = cv2.getTextSize(assoc_text, font, 0.38, 1)
            box_w = max(tw1, tw2) + 12
            box_h = th1 + th2 + 12

            label_y = max(y1 - box_h - 4, 10)
            cv2.rectangle(annotated, (x1, label_y), (x1 + box_w, label_y + box_h), color, -1)
            cv2.putText(annotated, header_text, (x1 + 6, label_y + th1 + 2), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, assoc_text, (x1 + 6, label_y + th1 + th2 + 6), font, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        return annotated
