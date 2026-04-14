import os
import json


class Checkpoint:
    """
    Handles saving and loading pipeline progress.

    Enables:
    - Resume from last processed timestamp
    - Fault tolerance for long videos
    """

    def __init__(self, video_path: str, checkpoint_dir: str = "output/checkpoints"):
        self.video_path = video_path

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Unique checkpoint file per video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{video_name}.json"
        )

    # ----------------------------
    # SAVE
    # ----------------------------

    def save(self, timestamp: float):
        """
        Save current progress.

        Args:
            timestamp (float): current processed timestamp (seconds)
        """

        data = {
            "last_timestamp": timestamp
        }

        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Checkpoint] Save failed: {e}")

    # ----------------------------
    # LOAD / RESUME
    # ----------------------------

    def resume_from(self) -> float:
        """
        Load last saved timestamp.

        Returns:
            float: timestamp to resume from
        """

        if not os.path.exists(self.checkpoint_path):
            print("[Checkpoint] No checkpoint found. Starting fresh.")
            return 0.0

        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)

            ts = data.get("last_timestamp", 0.0)

            print(f"[Checkpoint] Loaded checkpoint: {round(ts, 2)} sec")

            return ts

        except Exception as e:
            print(f"[Checkpoint] Load failed: {e}")
            return 0.0

    # ----------------------------
    # RESET
    # ----------------------------

    def reset(self):
        """
        Delete checkpoint file (start fresh run).
        """
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                print("[Checkpoint] Reset complete.")
        except Exception as e:
            print(f"[Checkpoint] Reset failed: {e}")