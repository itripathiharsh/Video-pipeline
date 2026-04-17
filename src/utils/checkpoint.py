import os
import json
import logging

# ----------------------------
# LOGGER SETUP
# ----------------------------
logger = logging.getLogger(__name__)


class Checkpoint:
    """
    Handles saving and loading pipeline progress.

    Enables:
    - Resume from last processed timestamp
    - Fault tolerance for long videos

    NOTE:
    - Uses /tmp (EC2-safe during runtime)
    - Uses S3-key-based naming to avoid collisions
    """

    def __init__(self, video_key: str, checkpoint_dir: str = "/tmp/checkpoints"):
        """
        Args:
            video_key (str): S3 key (NOT local path)
        """
        self.video_key = video_key

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 🔧 S3-SAFE UNIQUE NAME
        safe_name = video_key.replace("/", "_").replace(".mp4", "")

        self.checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{safe_name}.json"
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

        temp_path = self.checkpoint_path + ".tmp"

        try:
            # 🔧 ATOMIC WRITE (prevents corruption)
            with open(temp_path, "w") as f:
                json.dump(data, f)

            os.replace(temp_path, self.checkpoint_path)

        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

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
            logger.info("No checkpoint found. Starting fresh.")
            return 0.0

        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)

            ts = data.get("last_timestamp", 0.0)

            # 🔧 VALIDATION
            if not isinstance(ts, (int, float)):
                logger.warning("Invalid checkpoint value. Resetting.")
                return 0.0

            logger.info(f"Loaded checkpoint: {round(ts, 2)} sec")

            return float(ts)

        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
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
                logger.info("Checkpoint reset complete.")
        except Exception as e:
            logger.error(f"Checkpoint reset failed: {e}")