import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


class AudioVAD:
    """
    Audio activity scorer (NOT strict speech detection).

    Design:
    - Uses RMS energy as activity signal
    - No thresholds (continuous score)
    - Returns 0.0 if audio not available
    - Preloads audio once (efficient for long videos)
    """

    def __init__(self, audio_path: str = None):
        self.audio_path = audio_path
        self.available = False

        self.audio = None
        self.sr = None
        self.duration = 0.0

        if audio_path and librosa is not None:
            try:
                print(f"[AudioVAD] Loading audio: {audio_path}")

                self.audio, self.sr = librosa.load(
                    audio_path,
                    sr=16000,       # force 16kHz
                    mono=True
                )

                self.duration = len(self.audio) / self.sr
                self.available = True

                print(f"[AudioVAD] Loaded. Duration: {round(self.duration, 2)} sec")

            except Exception as e:
                print(f"[AudioVAD] Failed to load audio: {e}")
                self.available = False
        else:
            if librosa is None:
                print("[AudioVAD] librosa not installed. Audio disabled.")
            else:
                print("[AudioVAD] No audio provided.")

    def score_at(self, timestamp: float, window: float = 3.0) -> float:
        """
        Compute audio activity score at given timestamp.

        Args:
            timestamp (float): time in seconds
            window (float): analysis window (seconds)

        Returns:
            float: normalized audio activity score (0–1)
        """

        if not self.available:
            return 0.0

        start = int(timestamp * self.sr)
        end = int((timestamp + window) * self.sr)

        if start >= len(self.audio):
            return 0.0

        chunk = self.audio[start:end]

        if len(chunk) == 0:
            return 0.0

        # RMS energy
        rms = np.sqrt(np.mean(chunk ** 2))

        # Normalize (simple scaling)
        # Typical RMS is very small (~0.001–0.1), so scale up
        score = float(min(rms * 10.0, 1.0))

        return score

    def is_available(self) -> bool:
        """
        Returns whether audio is usable.
        """
        return self.available

    def reset(self):
        """
        Reset state (not required usually, but kept for consistency).
        """
        pass