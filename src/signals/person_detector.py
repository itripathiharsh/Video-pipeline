import torch
from ultralytics import YOLO


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
        self.model_path = cfg.YOLO_MODEL
        self.conf_threshold = cfg.YOLO_CONF

        self.device = self._get_device()

        print(f"[PersonDetector] Loading model: {self.model_path}")
        print(f"[PersonDetector] Using device: {self.device}")

        self.model = YOLO(self.model_path)

        # Move model to device explicitly
        if self.device == "cuda":
            self.model.to("cuda")
        else:
            self.model.to("cpu")

    def _get_device(self) -> str:
        """
        Select device automatically.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame):
        """
        Run person detection on a frame.

        Args:
            frame (np.ndarray): BGR image

        Returns:
            dict:
                {
                    "person_detected": bool,
                    "person_count": int
                }
        """

        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )

        person_count = 0

        for r in results:
            if r.boxes is None:
                continue

            classes = r.boxes.cls.cpu().numpy()

            for cls_id in classes:
                if int(cls_id) == self.PERSON_CLASS_ID:
                    person_count += 1

        return {
            "person_detected": person_count > 0,
            "person_count": person_count
        }