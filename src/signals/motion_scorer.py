import numpy as np
import cv2
import logging

# ----------------------------
# LOGGER SETUP
# ----------------------------
logger = logging.getLogger(__name__)


class MotionScorer:
    """
    Computes motion intensity between consecutive frames.

    Design Principles:
    - Lightweight (runs every frame)
    - No thresholds
    - No gating logic
    - Returns a continuous score (float)

    Output is later normalized inside SegmentBuilder.
    """

    def __init__(self):
        self.prev_gray = None

    def score(self, frame) -> float:
        """
        Compute motion score for current frame.
        """

        # 🔧 Handle None frame
        if frame is None:
            return 0.0

        # 🔧 Handle corrupt / unexpected shape
        if not hasattr(frame, "ndim") or frame.ndim != 3:
            return 0.0

        # Convert to grayscale
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return 0.0

        # First frame case
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Compute motion
        try:
            diff = cv2.absdiff(gray, self.prev_gray)
            score = float(np.mean(diff))
        except Exception:
            score = 0.0

        # 🔧 Optional debug (safe — not noisy unless enabled)
        logger.debug(f"Motion score: {score}")

        # Update previous frame
        self.prev_gray = gray

        return score

    def reset(self):
        """
        Reset internal state (useful if restarting stream).
        """
        self.prev_gray = None