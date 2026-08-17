# Multi-Camera Person Tracking & Re-Identification System

A multi-camera video analytics pipeline that detects people, tracks them within individual camera views, and associates the same person across different cameras using appearance-based re-identification.

The project is designed for the supplied three-video evaluation and includes duplicate-detection cleanup, track-level appearance aggregation, conservative cross-camera matching, identity auditing, and pipeline monitoring.

---

## What the System Does

```text
Camera Videos
     ↓
Person Detection
     ↓
Duplicate-Box Cleanup
     ↓
Local Person Tracking
     ↓
Stable Track Observations
     ↓
Global Re-ID Gallery
     ↓
Identity Decision
     ├── Confirmed Match
     ├── New Identity
     └── Needs Review
     ↓
Results + Logs + Dashboard
```

### Identity terminology

- `T1`, `T2`, etc. = camera-local tracker IDs.
- `Person_001`, `Person_002`, etc. = global identity IDs.
- A local tracker ID is not automatically a global identity.
- An uncertain `needs_review` result is never assigned the candidate's identity.

---

## Key Features

- Multi-camera video processing
- Person detection
- Duplicate/nested detection suppression
- Per-camera person tracking
- Appearance-based person re-identification
- Track-level appearance aggregation
- Multiple appearance prototypes per global identity
- Top-k prototype similarity for more stable matching
- Camera-aware Re-ID thresholds
- Conservative handling of uncertain matches
- Global identity management
- Identity audit logging
- Detection result storage
- Pipeline event logging
- Pipeline monitoring dashboard
- Compact event-based terminal output

---

## Supplied Evaluation Videos

The supplied videos are processed in sorted order:

```text
Camera 1 → foreground woman
Camera 2 → same foreground woman
Camera 3 → different foreground man
```

The expected identity relationship is:

```text
Person → Cameras

Same person → [Camera 1, Camera 2]
Different person → [Camera 3]
```

The exact numeric `Person_XXX` IDs may change between runs. The important result is the camera grouping.

---

## Re-Identification Logic

The Re-ID stage does not rely on a single frame.

For each stable track, multiple appearance observations are collected and used to build a more reliable representation.

Global identities maintain multiple appearance prototypes instead of relying on one initial frame.

The matching process uses the similarity of the best top-k prototypes rather than a single maximum similarity. This reduces accidental one-frame matches.

Same-camera identities are excluded from cross-camera matching.

If the similarity is not strong enough for a confirmed association, the system can return:

```text
needs_review
```

instead of forcing an incorrect identity.

---

## Camera-Specific Thresholds

The supplied videos have different viewpoints, so the current configuration uses camera-specific Re-ID thresholds:

```text
Camera 2 → 0.62
Camera 3 → 0.80
```

These values are calibrated for the supplied offline assessment videos. They should be recalibrated if different videos or camera setups are used.

---

## Project Structure

```text
multiple_camera_video_system/
│
├── main.py
│
├── core/
│   ├── pipeline.py
│   ├── detector.py
│   └── detection_worker.py
│
├── config/
│   └── config.json
│
├── videos/
│   ├── camera video 1
│   ├── camera video 2
│   └── camera video 3
│
├── results/
│   ├── detections/
│   ├── identity_audit.jsonl
│   ├── identity_summary.json
│   ├── pipeline_events.jsonl
│   └── pipeline_monitoring_dashboard.png
│
├── requirements.txt
└── README.md
```

> The `annotated/` output folder is intentionally not created by the project.

---

## Output Files

After execution, results are stored in:

```text
results/
```

### Detection Results

```text
results/detections/
```

Contains representative detection images generated during processing.

Example:

```text
cam2_Person_001_frame00080.jpg
```

### Identity Audit

```text
results/identity_audit.jsonl
```

Stores identity-related decisions and matching information for auditing.

### Identity Summary

```text
results/identity_summary.json
```

Contains the final global identity summary, including camera associations, appearances, similarity information, and review information.

### Pipeline Events

```text
results/pipeline_events.jsonl
```

Stores pipeline processing events in JSON Lines format.

### Monitoring Dashboard

```text
results/pipeline_monitoring_dashboard.png
```

Provides a visual summary of the pipeline execution and camera-wise processing statistics.

---

## Example Identity Summary

A successful run can produce information similar to:

```text
Person_001
    Cameras: [1, 2]
    Appearances: ...

Person_002
    Cameras: [3]
    Appearances: ...
```

The exact identity numbers and counts depend on the input videos and configuration.

---

## Example Terminal Summary

The pipeline ends with a compact summary similar to:

```text
[Worker] Done.

Processed sampled frames : ...
Detections                : ...
Tracked objects           : ...
Identity decisions        : ...
Cross-camera matches      : ...
Needs review              : ...
Known global identities   : ...
Processing speed          : ... FPS

============================================================
PERSON RE-ID SUMMARY
============================================================
...
```

The exact values depend on the input videos and runtime configuration.

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Place the input videos in the `videos/` directory and run:

```bash
python main.py
```

The pipeline automatically processes the available video sources and writes the results to:

```text
results/
```

---

## Configuration

Runtime settings are stored in:

```text
config/config.json
```

For a faster CPU test, the frame-processing budget can be reduced in the configuration file.

---

## Design Decisions

### Local Tracking and Global Identity Are Separate

A person can have different tracker IDs in different cameras:

```text
Camera 1 → Track 1
Camera 2 → Track 3
```

The Re-ID layer determines whether these observations should belong to the same global identity.

### Duplicate Detection Suppression

Almost identical or nested person detections are suppressed before tracking. This prevents one physical person from being unnecessarily represented by multiple local tracks.

### Track-Level Appearance Aggregation

Instead of making a global identity decision from a single frame, the system collects multiple stable observations from a track.

### Conservative Matching

A borderline similarity is not automatically treated as a confirmed match. Uncertain cases can be marked as `needs_review`.

This is especially important for preventing false cross-camera identity merges.

---

## Limitations

The quality of appearance-based Re-ID depends on:

- Camera viewpoint
- Lighting
- Occlusion
- Person visibility
- Video resolution
- Distance from the camera
- Clothing and appearance similarity
- Quality of the available appearance features

The current appearance encoder is the bundled ResNet18 ImageNet backbone from the original project rather than a specialized person-ReID model. Therefore, the corrected pipeline focuses on stable track aggregation, duplicate suppression, conservative matching, and dataset-specific calibration.

---

## Future Improvements

Possible extensions include:

- A dedicated person-ReID model
- Stronger appearance embeddings
- Temporal consistency for identity decisions
- Camera calibration and spatial constraints
- Improved occlusion handling
- GPU acceleration
- Real-time camera stream support
- Interactive review of `needs_review` cases
- Web-based monitoring

---

## Technologies

- Python
- OpenCV
- Computer Vision
- Person Detection
- Object Tracking
- Appearance-based Re-ID
- JSON / JSONL logging
- Data Visualization

---

## Summary

This project provides a modular multi-camera person tracking and re-identification pipeline.

The system:

1. Reads multiple camera videos.
2. Detects people.
3. Removes duplicate or nested detections.
4. Tracks people independently within each camera.
5. Collects stable appearance observations.
6. Compares observations across cameras.
7. Creates or updates global identities using conservative matching.
8. Separates uncertain matches as `needs_review`.
9. Generates detection results, identity logs, event logs, and a monitoring dashboard.

The design keeps camera-level tracking separate from global identity management, making the pipeline easier to extend to additional cameras and larger deployments.
