import logging
from typing import List, Dict

# ----------------------------
# LOGGER SETUP
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SegmentMerger] %(message)s"
)
logger = logging.getLogger(__name__)


class SegmentMerger:
    """
    Post-processing step to merge short segments.

    Goal:
    - Ensure clips are ~60–120 sec
    - Preserve semantic continuity
    - NEVER force artificial padding
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # 🔧 CONFIG-DRIVEN (fallback safe)
        self.target_min = getattr(cfg, "TARGET_CLIP_MIN", 60.0)
        self.soft_min = getattr(cfg, "SOFT_CLIP_MIN", 45.0)

    def merge(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge short segments intelligently.
        """

        if not segments:
            return []

        logger.info(f"Merging {len(segments)} segments...")

        # 🔧 VALIDATION (production safety)
        required_keys = ["start", "end", "duration"]
        for seg in segments:
            for key in required_keys:
                if key not in seg:
                    raise ValueError(f"Missing key '{key}' in segment: {seg}")

        merged = []
        i = 0

        while i < len(segments):
            curr = segments[i]

            # If already large enough → keep
            if curr["duration"] >= self.target_min:
                merged.append(curr)
                i += 1
                continue

            # Try merging forward
            merged_segment = curr.copy()
            j = i + 1

            while j < len(segments):
                next_seg = segments[j]

                gap = next_seg["start"] - merged_segment["end"]

                # Only merge if gap is small (continuity preserved)
                if gap > self.cfg.BRIDGE_GAP:
                    break

                # Merge segments
                merged_segment["end"] = next_seg["end"]

                # 🔧 SAFE DURATION (no negatives)
                merged_segment["duration"] = max(
                    merged_segment["end"] - merged_segment["start"],
                    0.0
                )

                # Combine metrics (weighted average)
                merged_segment = self._merge_metrics(
                    merged_segment,
                    next_seg
                )

                if merged_segment["duration"] >= self.target_min:
                    break

                j += 1

            # If still too small, allow soft minimum
            if merged_segment["duration"] < self.soft_min:
                merged.append(merged_segment)
                i = j
                continue

            merged.append(merged_segment)
            i = j

        # 🔧 CLEAN CLIP IDS
        for idx, seg in enumerate(merged):
            seg["clip_id"] = idx
            seg["prev_clip_id"] = idx - 1 if idx > 0 else None
            seg["next_clip_id"] = idx + 1 if idx < len(merged) - 1 else None

        logger.info(f"Final merged segments: {len(merged)}")

        return merged

    def _merge_metrics(self, seg1: Dict, seg2: Dict) -> Dict:
        """
        Combine metrics of two segments using weighted average.
        """

        total_duration = seg1["duration"] + seg2["duration"]

        def weighted_avg(v1, d1, v2, d2):
            return (v1 * d1 + v2 * d2) / max(total_duration, 1e-6)

        seg1["person_ratio"] = weighted_avg(
            seg1.get("person_ratio", 0.0), seg1["duration"],
            seg2.get("person_ratio", 0.0), seg2["duration"]
        )

        seg1["motion_score"] = weighted_avg(
            seg1.get("motion_score", 0.0), seg1["duration"],
            seg2.get("motion_score", 0.0), seg2["duration"]
        )

        seg1["audio_score"] = weighted_avg(
            seg1.get("audio_score", 0.0), seg1["duration"],
            seg2.get("audio_score", 0.0), seg2["duration"]
        )

        seg1["meaningfulness_score"] = weighted_avg(
            seg1.get("meaningfulness_score", 0.0), seg1["duration"],
            seg2.get("meaningfulness_score", 0.0), seg2["duration"]
        )

        seg1["merged"] = True

        return seg1