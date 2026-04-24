```python
import os
import shutil
import json
import time
import logging
import boto3

# Prevent thread explosion (important for EC2 stability)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from config import Config

# Ingestion
from src.ingestion.ffmpeg_stream import FFmpegFrameStreamer

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
from src.storage.s3_storage import S3Storage

# Utils
from src.utils.checkpoint import Checkpoint

# ----------------------------
# LOGGER
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# 🔹 PROCESS SINGLE VIDEO (CORE PIPELINE)
# =========================================================
def process_single_video(s3_key, cfg, s3):
    try:
        # ----------------------------
        # PARSE METADATA
        # ----------------------------
        school_name, date_folder = s3.parse_s3_key(s3_key)

        logger.info(f"Processing: {s3_key}")

        # ----------------------------
        # SAFE LOCAL PATH
        # ----------------------------
        safe_name = s3_key.replace("/", "_")
        local_video_path = f"/tmp/{safe_name}.mp4"

        # ----------------------------
        # DOWNLOAD VIDEO
        # ----------------------------
        s3.download_video(s3_key, local_video_path)

        # ----------------------------
        # PATHS (EC2 SAFE)
        # ----------------------------
        VIDEO_PATH = local_video_path
        OUTPUT_DIR = "/tmp/output"

        # Clean previous output
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
        AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

        os.makedirs(CLIPS_DIR, exist_ok=True)
        os.makedirs(AUDIO_DIR, exist_ok=True)

        logger.info("Starting pipeline...")

        # ----------------------------
        # AUDIO (DISABLED - LIGHT MODE)
        # ----------------------------
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

        # ----------------------------
        # CHECKPOINT
        # ----------------------------
        checkpoint = Checkpoint(s3_key)

        last_processed_ts = checkpoint.resume_from()
        last_yolo_ts = last_processed_ts
        last_detection = False

        logger.info(f"Resuming from: {round(last_processed_ts, 2)} sec")

        # ----------------------------
        # MAIN FRAME LOOP
        # ----------------------------
        for frame, timestamp in streamer.frames():

            if timestamp < last_processed_ts:
                continue

            # YOLO INTERVAL CONTROL
            person_detected = last_detection

            if (timestamp - last_yolo_ts) >= cfg.YOLO_INTERVAL:
                result = detector.detect(frame)
                person_detected = result["person_detected"]
                last_detection = person_detected
                last_yolo_ts = timestamp

            # MOTION
            motion_score = motion.score(frame)

            # AUDIO
            audio_score = audio_vad.score_at(timestamp)

            # SEGMENT BUILDING
            segment_builder.process(
                timestamp=timestamp,
                person_detected=person_detected,
                motion_score=motion_score,
                audio_score=audio_score
            )

            # SAVE CHECKPOINT EVERY 10s
            if int(timestamp) % 10 == 0:
                checkpoint.save(timestamp)

        logger.info("Frame processing complete.")

        # ----------------------------
        # FINALIZE SEGMENTS
        # ----------------------------
        segments = segment_builder.finalize()
        logger.info(f"Raw segments: {len(segments)}")

        segments = merger.merge(segments)
        logger.info(f"Merged segments: {len(segments)}")

        # ----------------------------
        # EXTRACT CLIPS
        # ----------------------------
        final_metadata = extractor.extract_all(segments)

        # ----------------------------
        # SAVE METADATA LOCALLY
        # ----------------------------
        metadata_path = storage.save_metadata(final_metadata)

        # ----------------------------
        # UPLOAD CLIPS TO S3
        # ----------------------------
        clip_paths = [seg["local_path"] for seg in final_metadata]

        s3.upload_clips(
            local_clip_paths=clip_paths,
            school_name=school_name,
            date_folder=date_folder
        )

        # ----------------------------
        # UPLOAD METADATA TO S3
        # ----------------------------
        try:
            s3.s3.upload_file(
                metadata_path,
                cfg.S3_BUCKET,
                f"input_video/{school_name}/{date_folder}/metadata.json"
            )
        except Exception as e:
            logger.error(f"Metadata upload failed: {e}")

        logger.info(f"Completed processing for {school_name}\n")

    except Exception as e:
        logger.error(f"Pipeline failed for {s3_key}: {e}")


# =========================================================
# 🔹 SQS WORKER LOOP (EVENT-DRIVEN)
# =========================================================
def run_worker():
    cfg = Config()
    s3 = S3Storage(bucket_name=cfg.S3_BUCKET)

    sqs = boto3.client("sqs", region_name=cfg.AWS_DEFAULT_REGION)

    # ⚠️ REPLACE THIS WITH YOUR ACTUAL QUEUE URL
    QUEUE_URL = "YOUR_SQS_QUEUE_URL"

    logger.info("Worker started. Waiting for messages...")

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20  # long polling
            )

            messages = response.get("Messages", [])

            if not messages:
                continue

            for msg in messages:
                try:
                    body = json.loads(msg["Body"])

                    # Extract S3 key from event
                    s3_key = body["Records"][0]["s3"]["object"]["key"]

                    logger.info(f"Received job: {s3_key}")

                    # Process video
                    process_single_video(s3_key, cfg, s3)

                    # Delete message after success
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=msg["ReceiptHandle"]
                    )

                    logger.info(f"Deleted message for: {s3_key}")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"SQS polling error: {e}")
            time.sleep(5)  # small backoff


# =========================================================
# 🔹 ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_worker()
```
