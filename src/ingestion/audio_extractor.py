import os
import subprocess


class AudioExtractor:
    """
    Extracts audio from video using FFmpeg.

    Features:
    - Converts to 16kHz mono WAV (best for analysis)
    - Skips extraction if already exists
    - Works efficiently for long videos
    """

    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.audio_path = os.path.join(self.output_dir, f"{base_name}.wav")

    # ----------------------------
    # AUDIO EXTRACTION
    # ----------------------------

    def extract(self, force: bool = False) -> str:
        """
        Extract audio from video.

        Args:
            force (bool): overwrite existing audio if True

        Returns:
            str: path to audio file or None if failed
        """

        # Skip if already exists
        if os.path.exists(self.audio_path) and not force:
            print(f"[AudioExtractor] Using cached audio: {self.audio_path}")
            return self.audio_path

        print("[AudioExtractor] Extracting audio...")

        cmd = [
            "ffmpeg",
            "-y" if force else "-n",
            "-loglevel", "error",
            "-i", self.video_path,
            "-vn",                  # remove video
            "-acodec", "pcm_s16le", # WAV format
            "-ar", "16000",         # 16kHz
            "-ac", "1",             # mono
            self.audio_path
        ]

        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                print("[AudioExtractor] FFmpeg error:")
                print(process.stderr.decode())
                return None

            if not os.path.exists(self.audio_path):
                print("[AudioExtractor] Audio file not created.")
                return None

            print(f"[AudioExtractor] Audio saved: {self.audio_path}")
            return self.audio_path

        except Exception as e:
            print(f"[AudioExtractor] Exception: {e}")
            return None

    # ----------------------------
    # CHECK AUDIO PRESENCE
    # ----------------------------

    def has_audio(self) -> bool:
        """
        Check if video contains an audio stream.

        Returns:
            bool
        """

        cmd = [
            "ffprobe",
            "-i", self.video_path,
            "-show_streams",
            "-select_streams", "a",
            "-loglevel", "error"
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            output = result.stdout.decode()

            return "codec_type=audio" in output

        except Exception:
            return False