import torch
import numpy as np
import logging
from ultralytics import YOLO

# ----------------------------
# LOGGER SETUP
# ----------------------------
logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Person detection using YOLOv8.

    Design:
    - Runs only when called (main controls interval)
    - Detects ONLY person class (class_id = 0)
    - Returns minimal info for segmentation
    """

    PERSON_CLASS_ID = 0  # COCO class for 'person'

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg.YOLO_MODEL
        self.conf_threshold = cfg.YOLO_CONF

        self.device = self._get_device()

        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

        self.model = YOLO(self.model_path)

        # Move model to device
        if self.device == "cuda":
            self.model.to("cuda")
        else:
            self.model.to("cpu")

        # 🔧 MODEL WARMUP (VERY IMPORTANT FOR EC2)
        try:
            dummy = np.zeros(
                (self.cfg.HEIGHT, self.cfg.WIDTH, 3),
                dtype=np.uint8
            )
            self.model(dummy, verbose=False)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _get_device(self) -> str:
        """
        Select device automatically.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame):
        """
        Run person detection on a frame.

        Returns:
            dict:
                {
                    "person_detected": bool,
                    "person_count": int
                }
        """

        # 🔧 FRAME SAFETY
        if frame is None:
            return {"person_detected": False, "person_count": 0}

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"person_detected": False, "person_count": 0}

        person_count = 0

        for r in results:
            if r.boxes is None:
                continue

            # 🔧 MICRO-OPTIMIZATION
            classes = r.boxes.cls.tolist()

            for cls_id in classes:
                if int(cls_id) == self.PERSON_CLASS_ID:
                    person_count += 1

        return {
            "person_detected": person_count > 0,
            "person_count": person_count
        }