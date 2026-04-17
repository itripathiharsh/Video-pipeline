import subprocess
import numpy as np
import cv2
import logging
from typing import Generator, Tuple

# ----------------------------
# LOGGER SETUP
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FFmpegFrameStreamer] %(message)s"
)
logger = logging.getLogger(__name__)


class FFmpegFrameStreamer:
    """
    Streams frames from video using FFmpeg without saving to disk.

    Key Features:
    - Constant memory usage (no frame dumping)
    - Works for long videos (8–10 hrs+)
    - Provides accurate timestamps
    - Downsamples FPS for efficiency
    """

    def __init__(self, video_path: str, cfg):
        self.video_path = video_path
        self.fps = cfg.FPS
        self.width = cfg.WIDTH
        self.height = cfg.HEIGHT

        self.process = None

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------

    def _build_command(self):
        """
        FFmpeg command to stream raw frames.
        """
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-i", self.video_path,
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]

    def _start(self):
        """
        Start FFmpeg subprocess.
        """
        logger.info(f"Starting stream: {self.video_path}")

        cmd = self._build_command()

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def _frame_generator(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Core generator logic (used internally with retry wrapper).
        """
        if self.process is None:
            self._start()

        frame_size = self.width * self.height * 3  # BGR
        frame_count = 0

        while True:
            raw = self.process.stdout.read(frame_size)

            if len(raw) != frame_size:
                logger.warning("FFmpeg stream ended unexpectedly or completed.")
                break

            frame = np.frombuffer(raw, np.uint8).reshape(
                (self.height, self.width, 3)
            )

            timestamp = frame_count / self.fps

            yield frame, timestamp

            frame_count += 1

        # 🔧 Detect early crash
        if self.process and self.process.poll() is not None:
            logger.warning("FFmpeg process exited early.")

    # ----------------------------
    # PUBLIC API WITH RETRY
    # ----------------------------

    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator yielding (frame, timestamp) with retry mechanism.
        """
        retry = 1  # minimal retry (can increase later)

        for attempt in range(retry + 1):
            try:
                yield from self._frame_generator()
                return

            except Exception as e:
                logger.error(f"FFmpeg failed (attempt {attempt}): {e}")
                self.close()

                if attempt == retry:
                    logger.error("FFmpeg stream crashed permanently.")
                    raise

                logger.info("Retrying FFmpeg stream...")

    # ----------------------------
    # CLEANUP
    # ----------------------------

    def close(self):
        """
        Cleanly terminate FFmpeg process.
        """
        if self.process:
            try:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

                self.process.kill()

            except Exception as e:
                logger.warning(f"Error while closing FFmpeg process: {e}")

            finally:
                self.process = None