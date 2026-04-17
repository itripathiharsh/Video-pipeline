class Config:
    """
    Central configuration for the video processing pipeline.
    Cloud-ready (S3 + EC2).
    """

    # ----------------------------
    # S3 CONFIG
    # ----------------------------

    S3_BUCKET = "your-bucket-name"   # 🔥 MUST CHANGE

    # ----------------------------
    # INGESTION (FRAME SAMPLING)
    # ----------------------------

    FPS = 1
    WIDTH = 480
    HEIGHT = 270

    # ----------------------------
    # PERSON DETECTION (YOLO)
    # ----------------------------

    YOLO_MODEL = "yolov8n.pt"
    YOLO_INTERVAL = 3.0
    YOLO_CONF = 0.35

    # ----------------------------
    # SEGMENTATION RULES
    # ----------------------------

    MIN_DURATION = 20.0
    MAX_DURATION = 120.0
    HARD_LIMIT = 150.0
    BRIDGE_GAP = 15.0

    # ----------------------------
    # MERGING (POST-PROCESSING)
    # ----------------------------

    TARGET_CLIP_MIN = 60.0     # 🔧 FIXED NAME
    SOFT_CLIP_MIN = 45.0       # 🔧 FIXED NAME

    # ----------------------------
    # SCORING WEIGHTS
    # ----------------------------

    WEIGHT_PERSON = 0.6
    WEIGHT_MOTION = 0.25
    WEIGHT_AUDIO = 0.15

    MIN_MEANINGFULNESS = 0.10

    # ----------------------------
    # PERFORMANCE / SAFETY
    # ----------------------------

    CHECKPOINT_DIR = "/tmp/checkpoints"   # 🔧 EC2 SAFE
    AUDIO_DIR = "/tmp/audio"
    CLIPS_DIR = "/tmp/clips"

    # ----------------------------
    # DEBUG / LOGGING
    # ----------------------------

    PRINT_PROGRESS = True