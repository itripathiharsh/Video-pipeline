from typing import List, Dict


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
        self.target_min = 60.0          # desired minimum
        self.soft_min = 45.0            # acceptable minimum if merge not possible

    def merge(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge short segments intelligently.

        Args:
            segments: raw segments from SegmentBuilder

        Returns:
            merged segments
        """

        if not segments:
            return []

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
                merged_segment["duration"] = (
                    merged_segment["end"] - merged_segment["start"]
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
                # keep anyway (do NOT drop data)
                merged.append(merged_segment)
                i = j
                continue

            merged.append(merged_segment)
            i = j

        # Reassign clip IDs cleanly
        for idx, seg in enumerate(merged):
            seg["clip_id"] = idx
            seg["prev_clip_id"] = idx - 1 if idx > 0 else None
            seg["next_clip_id"] = idx + 1 if idx < len(merged) - 1 else None

        return merged

    def _merge_metrics(self, seg1: Dict, seg2: Dict) -> Dict:
        """
        Combine metrics of two segments using weighted average.
        """

        total_duration = seg1["duration"] + seg2["duration"]

        def weighted_avg(v1, d1, v2, d2):
            return (v1 * d1 + v2 * d2) / max(total_duration, 1e-6)

        seg1["person_ratio"] = weighted_avg(
            seg1["person_ratio"], seg1["duration"],
            seg2["person_ratio"], seg2["duration"]
        )

        seg1["motion_score"] = weighted_avg(
            seg1["motion_score"], seg1["duration"],
            seg2["motion_score"], seg2["duration"]
        )

        seg1["audio_score"] = weighted_avg(
            seg1["audio_score"], seg1["duration"],
            seg2["audio_score"], seg2["duration"]
        )

        seg1["meaningfulness_score"] = weighted_avg(
            seg1["meaningfulness_score"], seg1["duration"],
            seg2["meaningfulness_score"], seg2["duration"]
        )

        seg1["merged"] = True

        return seg1