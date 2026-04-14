class Config:
    """
    Central configuration for the video processing pipeline.
    Modify values here to tune performance, accuracy, and behavior.
    """

    # ----------------------------
    # INPUT / OUTPUT
    # ----------------------------

    VIDEO_PATH = "data/input/video.mp4"
    OUTPUT_DIR = "output"

    # ----------------------------
    # INGESTION (FRAME SAMPLING)
    # ----------------------------

    FPS = 1                 # frames per second to process
    WIDTH = 480              # resize width
    HEIGHT = 270             # resize height

    # ----------------------------
    # PERSON DETECTION (YOLO)
    # ----------------------------

    YOLO_MODEL = "yolov8n.pt"   # nano model (fastest)
    YOLO_INTERVAL = 3.0         # run detection every N seconds
    YOLO_CONF = 0.35            # confidence threshold

    # ----------------------------
    # SEGMENTATION RULES
    # ----------------------------

    MIN_DURATION = 20.0         # drop very short noise clips
    MAX_DURATION = 120.0        # soft split (ideal max)
    HARD_LIMIT = 150.0          # hard cap (never exceed)
    BRIDGE_GAP = 15.0           # allowed gap without person

    # ----------------------------
    # MERGING (POST-PROCESSING)
    # ----------------------------

    TARGET_MIN_DURATION = 60.0  # desired final minimum clip length
    SOFT_MIN_DURATION = 45.0    # acceptable fallback minimum

    # ----------------------------
    # SCORING WEIGHTS
    # ----------------------------

    WEIGHT_PERSON = 0.6         # most important
    WEIGHT_MOTION = 0.25        # secondary signal
    WEIGHT_AUDIO = 0.15         # tertiary (often 0 for CCTV)

    MIN_MEANINGFULNESS = 0.10   # drop low-quality segments

    # ----------------------------
    # PERFORMANCE / SAFETY
    # ----------------------------

    CHECKPOINT_DIR = "output/checkpoints"
    AUDIO_DIR = "output/audio"
    CLIPS_DIR = "output/clips"

    # ----------------------------
    # DEBUG / LOGGING
    # ----------------------------

    PRINT_PROGRESS = True