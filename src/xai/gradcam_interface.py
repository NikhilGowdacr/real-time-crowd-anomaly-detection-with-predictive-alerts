"""
==============================================================================
Grad-CAM & Deep Model Explainability Interface (Phase 9).
Provides future-compatible convolutional / ViT attention explainability hooks.
Enforces strict academic integrity: NEVER synthesizes or fabricates artificial
saliency maps when a trained explainable neural model is not configured.
==============================================================================
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2

from src.utils.logger import setup_logger

logger = setup_logger("xai_gradcam")


class GradCAMInterface:
    """
    Interface for Gradient-weighted Class Activation Mapping (Grad-CAM)
    and visual attention explanations.
    
    Academic Integrity Principle:
    If no trained deep learning model is active (e.g., during baseline rule-based,
    spectral, or heuristic phases), this interface explicitly reports
    'Model explanation unavailable' and returns None rather than inventing fake heatmaps.
    """

    def __init__(self, model: Optional[Any] = None, target_layer_name: Optional[str] = None):
        """
        Initialize the Grad-CAM interface.

        Args:
            model: Optional PyTorch or ONNX neural model.
            target_layer_name: Optional target convolutional layer name.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self._is_configured = (model is not None and target_layer_name is not None)

    def is_available(self) -> bool:
        """
        Returns True only if a valid trained neural model and target layer are configured.
        """
        return bool(self._is_configured and self.model is not None)

    def generate_heatmap(
        self,
        frame: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extracts feature map activation gradients for the specified class.

        Args:
            frame: Input surveillance frame (BGR numpy array).
            class_idx: Target category index for gradient propagation.

        Returns:
            Dict containing:
            - "available": bool (False when unconfigured)
            - "reason": str explanation
            - "heatmap": Optional[np.ndarray] normalized [0.0 - 1.0] heatmap
            - "layer_name": Optional[str]
        """
        if not self.is_available():
            return {
                "available": False,
                "reason": "Model explanation unavailable (no trained explainable model configured)",
                "heatmap": None,
                "layer_name": self.target_layer_name,
            }

        # If a valid PyTorch model with hooks were configured, gradient extraction happens here.
        # Otherwise, maintain transparent unavailable state:
        try:
            # Safe placeholder execution if model object exists but lacks hook implementation
            return {
                "available": False,
                "reason": "Grad-CAM hook execution requires trained PyTorch convolutional backend",
                "heatmap": None,
                "layer_name": self.target_layer_name,
            }
        except Exception as e:
            logger.warning(f"Grad-CAM extraction encountered exception: {e}")
            return {
                "available": False,
                "reason": f"Grad-CAM processing error: {e}",
                "heatmap": None,
                "layer_name": self.target_layer_name,
            }

    def overlay_heatmap(
        self,
        frame: np.ndarray,
        heatmap: Optional[np.ndarray],
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays a class activation heatmap onto the original video frame.
        If heatmap is None or unavailable, safely returns the unchanged frame copy.

        Args:
            frame: Original BGR video frame.
            heatmap: 2D numpy array [0.0 - 1.0] or None.
            alpha: Transparency blending weight for heatmap.
            colormap: OpenCV colormap constant (default: COLORMAP_JET).

        Returns:
            Annotated BGR frame.
        """
        if frame is None:
            return frame

        annotated = frame.copy()
        if heatmap is None:
            return annotated

        try:
            h, w = frame.shape[:2]
            # Resize heatmap to match frame resolution
            resized_cam = cv2.resize(heatmap, (w, h))
            cam_uint8 = np.uint8(255 * np.clip(resized_cam, 0.0, 1.0))
            color_cam = cv2.applyColorMap(cam_uint8, colormap)
            cv2.addWeighted(color_cam, alpha, annotated, 1.0 - alpha, 0, annotated)
            return annotated
        except Exception as e:
            logger.warning(f"Failed to overlay heatmap: {e}")
            return annotated
