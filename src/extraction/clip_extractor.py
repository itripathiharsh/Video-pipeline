import os
import subprocess
import logging
from typing import List, Dict

# ----------------------------
# LOGGER SETUP (EC2 SAFE)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ClipExtractor] %(message)s"
)
logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Extracts video clips from source video using FFmpeg.

    Features:
    - Accurate timestamp-based clipping
    - Fast path using stream copy
    - Fallback to re-encoding if needed
    - Works for long videos
    - Fault-tolerant (continues on failure)
    """

    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = video_path

        # 🔧 EC2 SAFE TEMP DIRECTORY
        self.output_dir = output_dir or "/tmp/clips"
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------
    # HELPERS
    # ----------------------------

    def _format_time(self, seconds: float) -> str:
        """
        Convert seconds → HH:MM:SS.mmm format (FFmpeg compatible)
        """
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60

        return f"{hrs:02}:{mins:02}:{secs:06.3f}"

    def _build_output_path(self, clip_id: int) -> str:
        return os.path.join(self.output_dir, f"clip_{clip_id:05}.mp4")

    # ----------------------------
    # CORE EXTRACTION
    # ----------------------------

    def extract_clip(self, start: float, end: float, clip_id: int) -> str:
        """
        Extract a single clip using FFmpeg.

        Strategy:
        1. Try fast copy mode
        2. If fails → fallback to re-encoding
        """

        duration = max(end - start, 0.1)
        output_path = self._build_output_path(clip_id)

        start_str = self._format_time(start)
        duration_str = self._format_time(duration)

        # -------- Attempt 1: Fast Copy --------
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", self.video_path,
            "-ss", start_str,
            "-t", duration_str,
            "-c", "copy",
            output_path
        ]

        try:
            process = subprocess.run(
                cmd_copy,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode == 0 and os.path.exists(output_path):
                return output_path

            # -------- Attempt 2: Re-encode Fallback --------
            logger.warning(f"Copy failed, retrying (clip {clip_id})")

            cmd_encode = [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", self.video_path,
                "-ss", start_str,
                "-t", duration_str,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                output_path
            ]

            process = subprocess.run(
                cmd_encode,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                logger.error(f"Failed clip {clip_id}")
                logger.error(process.stderr.decode())
                return None

            if not os.path.exists(output_path):
                logger.error(f"Output missing for clip {clip_id}")
                return None

            return output_path

        except Exception as e:
            logger.exception(f"Exception for clip {clip_id}: {e}")
            return None

    # ----------------------------
    # BULK EXTRACTION
    # ----------------------------

    def extract_all(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract all clips from segment list.

        Args:
            segments: list of segment metadata

        Returns:
            list: updated metadata with file paths
        """

        logger.info(f"Extracting {len(segments)} clips...")

        final_metadata = []

        for seg in segments:
            clip_id = seg["clip_id"]

            output_path = self.extract_clip(
                start=seg["start"],
                end=seg["end"],
                clip_id=clip_id
            )

            if output_path is None:
                continue  # skip failed clip

            # 🔧 S3 + PIPELINE READY METADATA
            seg["local_path"] = output_path
            seg["file_name"] = os.path.basename(output_path)

            final_metadata.append(seg)

        logger.info(f"Done. {len(final_metadata)} clips saved.")

        return final_metadata