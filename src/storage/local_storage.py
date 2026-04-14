import os
import json
from typing import List, Dict


class LocalStorage:
    """
    Handles local storage of:
    - Extracted clips
    - Metadata JSON

    Designed to be easily replaceable with S3Storage later.
    """

    def __init__(self, base_output_dir: str):
        self.base_dir = base_output_dir

        self.clips_dir = os.path.join(self.base_dir, "clips")
        self.meta_dir = os.path.join(self.base_dir, "metadata")

        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    # ----------------------------
    # CLIP PATH MANAGEMENT
    # ----------------------------

    def get_clip_path(self, clip_id: int) -> str:
        """
        Returns standardized path for clip file.
        """
        return os.path.join(self.clips_dir, f"clip_{clip_id:05}.mp4")

    # ----------------------------
    # METADATA STORAGE
    # ----------------------------

    def save_metadata(self, metadata: List[Dict], filename: str = "metadata.json") -> str:
        """
        Save metadata JSON.

        Args:
            metadata: list of segment data
            filename: metadata filename

        Returns:
            file path
        """

        path = os.path.join(self.meta_dir, filename)

        try:
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[LocalStorage] Metadata saved: {path}")
            return path

        except Exception as e:
            print(f"[LocalStorage] Failed to save metadata: {e}")
            return None

    # ----------------------------
    # OPTIONAL: SAVE INDIVIDUAL CLIP METADATA
    # ----------------------------

    def save_clip_metadata(self, clip_data: Dict):
        """
        Save metadata per clip (optional use).
        """

        clip_id = clip_data.get("clip_id", "unknown")
        path = os.path.join(self.meta_dir, f"clip_{clip_id:05}.json")

        try:
            with open(path, "w") as f:
                json.dump(clip_data, f, indent=2)
        except Exception as e:
            print(f"[LocalStorage] Failed clip metadata {clip_id}: {e}")

    # ----------------------------
    # FUTURE EXTENSION HOOK
    # ----------------------------

    def upload_placeholder(self, file_path: str) -> str:
        """
        Placeholder for future cloud upload.

        For now, returns local path.
        Later → replace with S3 URL.
        """
        return file_path