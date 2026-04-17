import os
import shutil
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging

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


def run_pipeline():
    cfg = Config()

    # ----------------------------
    # INIT S3
    # ----------------------------
    s3 = S3Storage(bucket_name=cfg.S3_BUCKET)

    # ----------------------------
    # FETCH SCHOOLS
    # ----------------------------
    schools = s3.list_schools()

    for school_prefix in schools:

        latest_video_key = s3.get_latest_video(school_prefix)

        if not latest_video_key:
            continue

        # ----------------------------
        # PARSE METADATA (FIXED ORDER)
        # ----------------------------
        school_name, date_folder = s3.parse_s3_key(latest_video_key)

        # ----------------------------
        # SKIP IF ALREADY PROCESSED (FIXED POSITION)
        # ----------------------------
        prefix = f"input_video/{school_name}/{date_folder}/"

        existing = s3.s3.list_objects_v2(
            Bucket=cfg.S3_BUCKET,
            Prefix=prefix
        )

        if "Contents" in existing:
            logger.info(f"Skipping already processed: {latest_video_key}")
            continue

        logger.info(f"Processing: {latest_video_key}")

        # ----------------------------
        # DOWNLOAD VIDEO (NOW SAFE)
        # ----------------------------
        local_video_path = f"/tmp/{school_name}_{date_folder}.mp4"
        s3.download_video(latest_video_key, local_video_path)

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
        # AUDIO (DISABLED)
        # ----------------------------
        logger.info("Audio disabled (light mode)")
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
        # CHECKPOINT (S3 KEY SAFE)
        # ----------------------------
        checkpoint = Checkpoint(latest_video_key)

        # ----------------------------
        # RESUME
        # ----------------------------
        last_processed_ts = checkpoint.resume_from()
        last_yolo_ts = last_processed_ts
        last_detection = False

        logger.info(f"Resuming from: {round(last_processed_ts, 2)} sec")

        # ----------------------------
        # MAIN LOOP
        # ----------------------------
        for frame, timestamp in streamer.frames():

            if timestamp < last_processed_ts:
                continue

            # YOLO INTERVAL
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

            # SEGMENT BUILD
            segment_builder.process(
                timestamp=timestamp,
                person_detected=person_detected,
                motion_score=motion_score,
                audio_score=audio_score
            )

            # CHECKPOINT (REDUCED FREQUENCY)
            if int(timestamp) % 10 == 0:
                checkpoint.save(timestamp)

        logger.info("Frame processing complete.")

        # ----------------------------
        # FINALIZE
        # ----------------------------
        segments = segment_builder.finalize()
        logger.info(f"Raw segments: {len(segments)}")

        segments = merger.merge(segments)
        logger.info(f"Merged segments: {len(segments)}")

        final_metadata = extractor.extract_all(segments)

        # ----------------------------
        # SAVE METADATA
        # ----------------------------
        metadata_path = storage.save_metadata(final_metadata)

        # ----------------------------
        # UPLOAD CLIPS
        # ----------------------------
        clip_paths = [seg["local_path"] for seg in final_metadata]

        s3.upload_clips(
            local_clip_paths=clip_paths,
            school_name=school_name,
            date_folder=date_folder
        )

        # ----------------------------
        # UPLOAD METADATA
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

    logger.info("ALL DONE 🚀")


if __name__ == "__main__":
    run_pipeline()