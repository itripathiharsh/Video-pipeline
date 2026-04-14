class SegmentBuilder:
    """
    Builds meaningful video segments based on:
    - Person presence (PRIMARY signal)
    - Motion score (secondary, scoring only)
    - Audio score (tertiary, scoring only)

    State Machine:
    IDLE → ACTIVE → BRIDGE → ACTIVE / CLOSE
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.current = None
        self.segments = []
        self.bridge_start = None

    def process(self, timestamp, person_detected,
                motion_score, audio_score):

        # ----------------------------
        # STATE: IDLE
        # ----------------------------
        if self.state == "IDLE":
            if person_detected:
                self._open(timestamp)

        # ----------------------------
        # STATE: ACTIVE
        # ----------------------------
        elif self.state == "ACTIVE":
            self._update(timestamp, person_detected,
                         motion_score, audio_score)

            duration = timestamp - self.current["start"]

            # HARD LIMIT (strict safety)
            if duration >= self.cfg.HARD_LIMIT:
                self._close("hard_split")
                return

            # SOFT LIMIT (normal split)
            if duration >= self.cfg.MAX_DURATION:
                self._close("soft_split")
                return

            # No person → go to bridge
            if not person_detected:
                self.state = "BRIDGE"
                self.bridge_start = timestamp

        # ----------------------------
        # STATE: BRIDGE
        # ----------------------------
        elif self.state == "BRIDGE":
            gap = timestamp - self.bridge_start

            if person_detected:
                # Person came back → continue segment
                self.state = "ACTIVE"
                self._update(timestamp, person_detected,
                             motion_score, audio_score)

            elif gap >= self.cfg.BRIDGE_GAP:
                # Too long without person → close
                self._close("gap_split")

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------

    def _open(self, timestamp):
        self.current = {
            "start": timestamp,
            "end": timestamp,
            "person_frames": 0,
            "total_frames": 0,
            "motion_scores": [],
            "audio_scores": []
        }
        self.state = "ACTIVE"

    def _update(self, timestamp, person_detected,
                motion_score, audio_score):

        seg = self.current

        seg["end"] = timestamp
        seg["total_frames"] += 1

        if person_detected:
            seg["person_frames"] += 1

        seg["motion_scores"].append(motion_score)
        seg["audio_scores"].append(audio_score)

    def _close(self, reason):
        seg = self.current

        duration = seg["end"] - seg["start"]

        # Drop too short segments (noise)
        if duration < self.cfg.MIN_DURATION:
            self._reset_current()
            return

        # ----------------------------
        # METRICS
        # ----------------------------
        total_frames = max(seg["total_frames"], 1)

        person_ratio = seg["person_frames"] / total_frames

        motion_avg = (
            sum(seg["motion_scores"]) /
            max(len(seg["motion_scores"]), 1)
        )

        audio_avg = (
            sum(seg["audio_scores"]) /
            max(len(seg["audio_scores"]), 1)
        )

        # ----------------------------
        # MOTION NORMALIZATION
        # (self-relative, prevents bias)
        # ----------------------------
        motion_ceiling = max(motion_avg * 3.0, 5.0)
        motion_norm = min(motion_avg / motion_ceiling, 1.0)

        # ----------------------------
        # FINAL SCORE
        # ----------------------------
        score = (
            self.cfg.WEIGHT_PERSON * person_ratio +
            self.cfg.WEIGHT_MOTION * motion_norm +
            self.cfg.WEIGHT_AUDIO  * audio_avg
        )

        # Drop low-quality segments
        if score < self.cfg.MIN_MEANINGFULNESS:
            self._reset_current()
            return

        # Save segment
        self.segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "duration": duration,
            "person_ratio": round(person_ratio, 4),
            "motion_score": round(motion_avg, 4),
            "audio_score": round(audio_avg, 4),
            "meaningfulness_score": round(score, 4),
            "end_reason": reason
        })

        self._reset_current()

    def _reset_current(self):
        self.current = None
        self.state = "IDLE"
        self.bridge_start = None

    # ----------------------------
    # FINALIZATION
    # ----------------------------

    def finalize(self):
        """
        Call after stream ends.
        """

        # Close any open segment
        if self.current:
            self._close("end_of_video")

        # Add linking metadata
        for i, seg in enumerate(self.segments):
            seg["clip_id"] = i
            seg["prev_clip_id"] = i - 1 if i > 0 else None
            seg["next_clip_id"] = i + 1 if i < len(self.segments) - 1 else None

        return self.segments