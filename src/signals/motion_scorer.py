import numpy as np
import cv2


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

        Args:
            frame (np.ndarray): BGR image

        Returns:
            float: motion score (unnormalized)
        """

        # Convert to grayscale (faster + enough for motion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame case
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Absolute difference between frames
        diff = cv2.absdiff(gray, self.prev_gray)

        # Mean intensity difference = motion magnitude
        score = float(np.mean(diff))

        # Update previous frame
        self.prev_gray = gray

        return score

    def reset(self):
        """
        Reset internal state (useful if restarting stream).
        """
        self.prev_gray = None