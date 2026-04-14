# CCTV Intelligent Pipeline

A sophisticated video segmentation and extraction system designed to intelligently process CCTV footage by analyzing multiple signals (person presence, motion, and audio activity) to identify and extract meaningful video segments. This pipeline transforms hours of raw security camera footage into focused, reviewable clips optimized for archival and analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Flow](#data-flow)
- [Algorithms and Formulas](#algorithms-and-formulas)
- [Output](#output)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Overview

The CCTV Intelligent Pipeline addresses the challenge of reviewing extensive CCTV footage by automatically identifying segments containing relevant activity. Instead of manually scanning hours of video, the system:

- Analyzes video streams using three independent signals: person detection, motion intensity, and audio activity
- Segments footage into meaningful clips based on activity patterns
- Merges short segments to create reviewable 60-120 second clips
- Extracts high-quality video clips without re-encoding when possible
- Provides comprehensive metadata for each extracted segment

This system is particularly valuable for security operations, surveillance review, and any scenario requiring efficient video content analysis.

## Features

- **Multi-Signal Analysis**: Combines person detection, motion scoring, and voice activity detection
- **Intelligent Segmentation**: State machine-based segmentation with bridging for temporary absences
- **Adaptive Scoring**: Weighted scoring system that adapts to content characteristics
- **Memory Efficient**: Processes arbitrarily long videos with constant memory usage
- **Fault Tolerant**: Checkpoint system for resumability and graceful error handling
- **Fast Processing**: Optimized for speed with interval-based detection and stream copying
- **Configurable**: Extensive configuration options for different use cases
- **Extensible**: Modular design allowing custom detectors and storage backends

## Architecture

The system is organized into seven major subsystems:

### 1. Ingestion Layer
- **FFmpegFrameStreamer**: Streams video frames directly from disk without buffering
- **AudioExtractor**: Extracts and caches audio streams for analysis

### 2. Signal Detection Layer
- **PersonDetector**: YOLOv8-based person detection (runs every 3 seconds)
- **MotionScorer**: Optical flow intensity calculation (runs every frame)
- **AudioVAD**: Voice Activity Detection via RMS energy analysis

### 3. Segmentation Engine
- **SegmentBuilder**: State machine that builds segments based on signal combinations

### 4. Post-Processing Layer
- **SegmentMerger**: Intelligently merges short segments to create optimal clip lengths

### 5. Extraction Layer
- **ClipExtractor**: FFmpeg-based clip extraction with fallback re-encoding

### 6. Storage Layer
- **LocalStorage**: Manages output files and metadata
- **S3Storage**: Optional cloud storage backend

### 7. Utility Layer
- **Checkpoint**: Fault tolerance and resumability
- **VideoUtils**: Video metadata extraction

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for faster YOLO inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cctv-intelligent-pipeline.git
cd cctv-intelligent-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model:
```bash
# The yolov8n.pt file should be included in the repository
# If not, download from ultralytics:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

4. Configure your environment (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Basic Usage

1. Place your input video in `data/input/` (default: `video.mp4`)

2. Run the pipeline:
```bash
python -m src.main
```

3. Find results in `data/output/`:
   - Extracted clips: `clips/`
   - Metadata: `metadata/metadata.json`
   - Checkpoints: `checkpoints/`

### Command Line Options

The system can be configured via `config.py` or environment variables. Key parameters include:

- Video input/output paths
- Detection intervals and thresholds
- Segmentation parameters
- Scoring weights

### Advanced Usage

For custom configurations, modify `config.py` or set environment variables:

```python
# Example: Change detection interval
export YOLO_INTERVAL=5.0
python -m src.main
```

## Configuration

All configuration is centralized in `config.py`. Key parameters:

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Input/Output** | `VIDEO_PATH` | `data/input/video.mp4` | Source video file |
| | `OUTPUT_DIR` | `output` | Results directory |
| **Ingestion** | `FPS` | 1 | Frame sampling rate |
| | `WIDTH`, `HEIGHT` | 480, 270 | Resize resolution |
| **Detection** | `YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| | `YOLO_INTERVAL` | 3.0 | Detection frequency (seconds) |
| | `YOLO_CONF` | 0.35 | Confidence threshold |
| **Segmentation** | `MIN_DURATION` | 20 | Minimum segment duration (seconds) |
| | `MAX_DURATION` | 120 | Soft split threshold (seconds) |
| | `HARD_LIMIT` | 150 | Absolute maximum duration (seconds) |
| | `BRIDGE_GAP` | 15 | Allowed absence gap (seconds) |
| **Scoring** | `WEIGHT_PERSON` | 0.6 | Person signal weight |
| | `WEIGHT_MOTION` | 0.25 | Motion signal weight |
| | `WEIGHT_AUDIO` | 0.15 | Audio signal weight |
| | `MIN_MEANINGFULNESS` | 0.1 | Quality threshold |
| **Merging** | `TARGET_MIN_DURATION` | 60 | Target clip length (seconds) |

## Data Flow

The pipeline processes video through these stages:

1. **Frame Streaming**: FFmpeg decompresses video at reduced FPS and resolution
2. **Signal Generation**: Three detectors analyze frames/audio independently
3. **Segmentation**: State machine builds segments based on activity patterns
4. **Post-Processing**: Short segments are merged to create optimal lengths
5. **Extraction**: FFmpeg extracts clips using stream copy when possible
6. **Storage**: Clips and metadata are saved with full traceability

## Algorithms and Formulas

### Motion Scoring
Calculates frame-to-frame intensity difference:
```
score = (1/(H×W)) × Σ|i,j| |I_t(i,j) - I_t-1(i,j)|
```

### Audio Scoring
RMS energy normalized to 0-1 range:
```
score = min(RMS × 10, 1.0)
```

### Segment Meaningfulness Score
Weighted combination of signals:
```
score = 0.6 × person_ratio + 0.25 × motion_norm + 0.15 × audio_avg
```

Where:
- `person_ratio` = fraction of frames with person detection
- `motion_norm` = normalized motion score (adaptive ceiling)
- `audio_avg` = average audio activity score

### Motion Normalization
Adaptive normalization based on content:
```
motion_norm = min(motion_avg / max(3 × motion_avg, 5), 1.0)
```

### Metric Merging
Weighted average by duration for merged segments:
```
merged_metric = (metric1 × duration1 + metric2 × duration2) / (duration1 + duration2)
```

## Output

The pipeline generates:

### Video Clips
- Location: `output/clips/clip_XXXXX.mp4`
- Format: H.264 video with AAC audio
- Length: 60-120 seconds (optimized for review)
- Quality: Stream copy when possible, re-encoded fallback

### Metadata
- Location: `output/metadata/metadata.json`
- Format: JSON array of segment objects
- Contents: Timestamps, scores, metrics, file paths, relationships

Example metadata entry:
```json
{
  "clip_id": 0,
  "start": 10.5,
  "end": 75.3,
  "duration": 64.8,
  "person_ratio": 0.95,
  "motion_score": 8.2,
  "audio_score": 0.25,
  "meaningfulness_score": 0.57,
  "end_reason": "soft_split",
  "file_path": "output/clips/clip_00000.mp4",
  "prev_clip_id": null,
  "next_clip_id": 1,
  "merged": false
}
```

### Checkpoints
- Location: `output/checkpoints/{video_name}.json`
- Purpose: Enable resumability after interruptions
- Contents: Last processed timestamp

## Technologies

- **FFmpeg**: Video/audio processing and extraction
- **OpenCV**: Image operations and motion detection
- **YOLOv8**: Person detection (ultralytics package)
- **librosa**: Audio analysis and loading
- **NumPy**: Numerical computations
- **PyTorch**: Deep learning framework (via ultralytics)
- **boto3**: AWS S3 integration (optional)
- **tqdm**: Progress visualization

## Performance Characteristics

- **Memory**: Constant usage (~50MB) regardless of video length
- **Speed**: ~200ms per YOLO inference, stream copy extraction
- **Scalability**: Handles 8-10 hour videos efficiently
- **Fault Tolerance**: Checkpoint-based resumability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

For questions or issues, please open a GitHub issue or contact the maintainers.