import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Central configuration for the video processing pipeline.
    Cloud-ready (S3 + EC2).
    """

    def __init__(self):

        # ----------------------------
        # ENV / AWS
        # ----------------------------
        self.ENV = os.getenv("ENV", "prod")
        self.AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self.S3_BUCKET = os.getenv("S3_BUCKET", "sentio-mind-storage")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.USE_GPU = os.getenv("USE_GPU", "false")

        # ----------------------------
        # INGESTION (FRAME SAMPLING)
        # ----------------------------
        self.FPS = 1
        self.WIDTH = 480
        self.HEIGHT = 270

        # ----------------------------
        # PERSON DETECTION (YOLO)
        # ----------------------------
        self.YOLO_MODEL = "yolov8n.pt"
        self.YOLO_INTERVAL = 3.0
        self.YOLO_CONF = 0.35

        # ----------------------------
        # SEGMENTATION RULES
        # ----------------------------
        self.MIN_DURATION = 20.0
        self.MAX_DURATION = 120.0
        self.HARD_LIMIT = 150.0
        self.BRIDGE_GAP = 15.0

        # ----------------------------
        # MERGING (POST-PROCESSING)
        # ----------------------------
        self.TARGET_CLIP_MIN = 60.0
        self.SOFT_CLIP_MIN = 45.0

        # ----------------------------
        # SCORING WEIGHTS
        # ----------------------------
        self.WEIGHT_PERSON = 0.6
        self.WEIGHT_MOTION = 0.25
        self.WEIGHT_AUDIO = 0.15

        self.MIN_MEANINGFULNESS = 0.10

        # ----------------------------
        # PERFORMANCE / PATHS (EC2 SAFE)
        # ----------------------------
        self.CHECKPOINT_DIR = "/tmp/checkpoints"
        self.AUDIO_DIR = "/tmp/audio"
        self.CLIPS_DIR = "/tmp/clips"

        # ----------------------------
        # DEBUG / LOGGING
        # ----------------------------
        self.PRINT_PROGRESS = True
