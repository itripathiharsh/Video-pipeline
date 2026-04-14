import subprocess
import json
import os


class VideoUtils:
    """
    Utility functions for video metadata using ffprobe.
    """

    @staticmethod
    def _run_ffprobe(cmd):
        """
        Run ffprobe command and return parsed JSON.
        """
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode != 0:
                print("[VideoUtils] ffprobe error:")
                print(result.stderr.decode())
                return None

            return json.loads(result.stdout.decode())

        except Exception as e:
            print(f"[VideoUtils] Exception: {e}")
            return None

    # ----------------------------
    # BASIC INFO
    # ----------------------------

    @staticmethod
    def get_duration(video_path: str) -> float:
        """
        Returns duration in seconds.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]

        data = VideoUtils._run_ffprobe(cmd)

        if not data:
            return 0.0

        try:
            return float(data["format"]["duration"])
        except Exception:
            return 0.0

    @staticmethod
    def get_fps(video_path: str) -> float:
        """
        Returns frames per second.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json",
            video_path
        ]

        data = VideoUtils._run_ffprobe(cmd)

        if not data:
            return 0.0

        try:
            fps_str = data["streams"][0]["r_frame_rate"]
            num, denom = map(int, fps_str.split("/"))
            return num / denom if denom != 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def get_resolution(video_path: str):
        """
        Returns (width, height)
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path
        ]

        data = VideoUtils._run_ffprobe(cmd)

        if not data:
            return None, None

        try:
            stream = data["streams"][0]
            return stream["width"], stream["height"]
        except Exception:
            return None, None

    # ----------------------------
    # VALIDATION
    # ----------------------------

    @staticmethod
    def is_valid_video(video_path: str) -> bool:
        """
        Checks if video file exists and is readable.
        """
        if not os.path.exists(video_path):
            print(f"[VideoUtils] File not found: {video_path}")
            return False

        duration = VideoUtils.get_duration(video_path)

        if duration <= 0:
            print("[VideoUtils] Invalid video (duration = 0)")
            return False

        return True

    # ----------------------------
    # DEBUG INFO
    # ----------------------------

    @staticmethod
    def print_video_info(video_path: str):
        """
        Print useful video information.
        """
        duration = VideoUtils.get_duration(video_path)
        fps = VideoUtils.get_fps(video_path)
        width, height = VideoUtils.get_resolution(video_path)

        print("\n[Video Info]")
        print(f"Path       : {video_path}")
        print(f"Duration   : {round(duration, 2)} sec")
        print(f"FPS        : {round(fps, 2)}")
        print(f"Resolution : {width} x {height}")
        print("")