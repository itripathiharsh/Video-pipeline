import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json

from config import Config

# Ingestion
from src.ingestion.ffmpeg_stream import FFmpegFrameStreamer
from src.ingestion.audio_extractor import AudioExtractor

# Signals
from src.signals.person_detector import PersonDetector
from src.signals.motion_scorer import MotionScorer
from src.signals.audio_vad import AudioVAD

# Segmentation
from src.segmentation.segment_builder import SegmentBuilder

# Post Processing
from src.post_processing.segment_merger import SegmentMerger

# Extraction
from src.extraction.clip_extractor import ClipExtractor

# Storage
from src.storage.local_storage import LocalStorage

# Utils
from src.utils.checkpoint import Checkpoint
# from src.utils.video_utils import VideoUtils


def run_pipeline():
    cfg = Config()

    # ----------------------------
    # PATHS
    # ----------------------------
    VIDEO_PATH = cfg.VIDEO_PATH
    OUTPUT_DIR = cfg.OUTPUT_DIR

    CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("\n[PIPELINE] Starting video processing...\n")

    # ----------------------------
    # VALIDATE VIDEO
    # ----------------------------
    # if not VideoUtils.is_valid_video(VIDEO_PATH):
    #     print("[PIPELINE] Invalid video. Exiting.")
    #     return

    # VideoUtils.print_video_info(VIDEO_PATH)

    # ----------------------------
    # AUDIO SETUP
    # ----------------------------
    # audio_extractor = AudioExtractor(VIDEO_PATH, AUDIO_DIR)

    # if audio_extractor.has_audio():
    #     audio_path = audio_extractor.extract()
    # else:
    #     print("[PIPELINE] No audio stream found.")
    #     audio_path = None

    # audio_vad = AudioVAD(audio_path)
    
    print("[PIPELINE] Audio disabled (light mode)")
    audio_vad = AudioVAD(None)

    # ----------------------------
    # INITIALIZE MODULES
    # ----------------------------
    streamer = FFmpegFrameStreamer(VIDEO_PATH, cfg)
    detector = PersonDetector(cfg)
    motion = MotionScorer()
    segment_builder = SegmentBuilder(cfg)
    merger = SegmentMerger(cfg)
    extractor = ClipExtractor(VIDEO_PATH, CLIPS_DIR)
    storage = LocalStorage(OUTPUT_DIR)
    checkpoint = Checkpoint(VIDEO_PATH)

    # ----------------------------
    # CHECKPOINT RESUME
    # ----------------------------
    last_processed_ts = checkpoint.resume_from()
    last_yolo_ts = last_processed_ts
    last_detection = False

    print(f"[PIPELINE] Resuming from: {round(last_processed_ts, 2)} sec")

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    for frame, timestamp in streamer.frames():

        # Skip already processed frames
        if timestamp < last_processed_ts:
            continue

        # ----------------------------
        # PERSON DETECTION (INTERVAL BASED)
        # ----------------------------
        person_detected = last_detection

        if (timestamp - last_yolo_ts) >= cfg.YOLO_INTERVAL:
            result = detector.detect(frame)

            person_detected = result["person_detected"]
            last_detection = person_detected
            last_yolo_ts = timestamp

        # ----------------------------
        # MOTION SCORE
        # ----------------------------
        motion_score = motion.score(frame)

        # ----------------------------
        # AUDIO SCORE
        # ----------------------------
        audio_score = audio_vad.score_at(timestamp)

        # ----------------------------
        # SEGMENT BUILDING
        # ----------------------------
        segment_builder.process(
            timestamp=timestamp,
            person_detected=person_detected,
            motion_score=motion_score,
            audio_score=audio_score
        )

        # ----------------------------
        # SAVE CHECKPOINT
        # ----------------------------
        checkpoint.save(timestamp)

    print("\n[PIPELINE] Frame processing complete.")

    # ----------------------------
    # FINALIZE SEGMENTS
    # ----------------------------
    segments = segment_builder.finalize()
    print(f"[PIPELINE] Raw segments: {len(segments)}")

    # ----------------------------
    # MERGE SEGMENTS (60 sec logic)
    # ----------------------------
    segments = merger.merge(segments)
    print(f"[PIPELINE] Segments after merging: {len(segments)}")

    # ----------------------------
    # EXTRACT CLIPS
    # ----------------------------
    final_metadata = extractor.extract_all(segments)

    # ----------------------------
    # SAVE METADATA
    # ----------------------------
    storage.save_metadata(final_metadata)

    print("\n[PIPELINE] DONE ✅\n")


if __name__ == "__main__":
    run_pipeline()