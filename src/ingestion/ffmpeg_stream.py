import subprocess
import numpy as np
import cv2
from typing import Generator, Tuple


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

    def _build_command(self):
        """
        FFmpeg command to stream raw frames.
        """
        return [
            "ffmpeg",
            "-loglevel", "error",              # suppress logs
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
        cmd = self._build_command()

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator yielding (frame, timestamp).

        Timestamp is computed based on FPS sampling.
        """
        if self.process is None:
            self._start()

        frame_size = self.width * self.height * 3  # BGR

        frame_count = 0

        try:
            while True:
                raw = self.process.stdout.read(frame_size)

                if len(raw) != frame_size:
                    break  # end of stream

                frame = np.frombuffer(raw, np.uint8).reshape(
                    (self.height, self.width, 3)
                )

                timestamp = frame_count / self.fps

                yield frame, timestamp

                frame_count += 1

        finally:
            self.close()

    def close(self):
        """
        Cleanly terminate FFmpeg process.
        """
        if self.process:
            try:
                self.process.stdout.close()
                self.process.stderr.close()
                self.process.kill()
            except Exception:
                pass
            finally:
                self.process = None