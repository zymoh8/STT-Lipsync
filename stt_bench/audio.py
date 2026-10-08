"""Read decoded samples, rather than feeding WAV container headers to models."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import wave


@dataclass(frozen=True)
class Audio:
    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2

    @property
    def duration_seconds(self) -> float:
        return len(self.pcm) / (self.sample_rate * self.channels * self.sample_width)

    def chunks(self, frames: int = 4000):
        if frames <= 0:
            raise ValueError("Chunk size must be positive")
        size = frames * self.channels * self.sample_width
        for offset in range(0, len(self.pcm), size):
            yield self.pcm[offset:offset + size]


def load_audio(path: str | Path, sample_rate: int = 16000) -> Audio:
    """Require uncompressed 16-bit mono WAV at the benchmark's chosen rate."""
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels() != 1 or source.getsampwidth() != 2
                or source.getcomptype() != "NONE" or source.getframerate() != sample_rate):
            raise ValueError(f"{path}: expected {sample_rate} Hz, 16-bit mono PCM WAV; use 'prepare'")
        frames = source.getnframes()
        pcm = source.readframes(frames)
        if not pcm or len(pcm) != frames * 2:
            raise ValueError(f"{path}: empty or truncated audio")
        return Audio(pcm, source.getframerate())


def prepare_audio(source: Path, destination: Path, sample_rate: int = 16000) -> None:
    """Use FFmpeg for decoding/downmixing/resampling; never overwrite input/output."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if not source.is_file():
        raise ValueError(f"Audio does not exist: {source}")
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source.resolve()),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        "-f", "wav", str(destination.resolve()),
    ], check=True)
    load_audio(destination, sample_rate)
