import json
import os
import queue
import shutil
import threading
from collections import defaultdict

from core.camera import camera_producer
from core.detector import detection_worker
from core.dashboard import create_dashboard
from core.logger import log_event


def _reset_output(config):
    output = config["output"]
    detections_dir = output.get("detections_dir", "results/detections")
    annotated_dir = output.get("annotated_video_dir", "results/annotated")
    if os.path.isdir(detections_dir):
        shutil.rmtree(detections_dir)
    if os.path.isdir(annotated_dir):
        shutil.rmtree(annotated_dir)

    for path in [
        output.get("log_file", "results/pipeline_events.jsonl"),
        output.get("identity_audit_file", "results/identity_audit.jsonl"),
        output.get("identity_summary_file", "results/identity_summary.json"),
        output.get("dashboard_file", "results/pipeline_monitoring_dashboard.png"),
    ]:
        if os.path.isfile(path):
            os.remove(path)

    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output.get("log_file", "results/pipeline_events.jsonl")) or ".", exist_ok=True)


def build_camera_data(log_file):
    camera_data = defaultdict(lambda: {
        "frames_pushed": 0,
        "frames_dropped": 0,
        "drop_rate": 0,
        "queue_fullness": [],
        "processing_lags": [],
        "detections": 0,
        "unique_tracks": 0,
        "active_tracks": 0,
    })

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            camera_id = event.get("camera_id")
            if camera_id is None:
                continue
            data = camera_data[camera_id]

            if "queue_fullness" in event:
                data["queue_fullness"].append(float(event["queue_fullness"]))
            if "processing_lag_sec" in event:
                data["processing_lags"].append(float(event["processing_lag_sec"]))
            if "frames_pushed" in event:
                data["frames_pushed"] = int(event["frames_pushed"])
            if "frames_dropped" in event:
                data["frames_dropped"] = int(event["frames_dropped"])
            if "frame_drop_rate" in event:
                data["drop_rate"] = float(event["frame_drop_rate"])
            if event.get("level") == "EVENT":
                data["detections"] += len(event.get("detections", []))
                track_ids = [
                    t.get("track_id")
                    for t in event.get("tracks", [])
                    if t.get("track_id") is not None
                ]
                if track_ids:
                    data["active_tracks"] = len(track_ids)
                    data["unique_tracks"] = max(data["unique_tracks"], max(track_ids))

    return dict(camera_data)


def build_identity_summary(log_file, output_file):
    identities = defaultdict(lambda: {
        "cameras": set(),
        "appearances": 0,
        "best_similarity": 0.0,
        "matches": [],
        "review_count": 0,
        "new_identity_count": 0,
        "cross_camera_match_count": 0,
    })

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("level") != "IDENTITY_MATCH":
                continue

            identity = event.get("identity")
            if not identity:
                continue

            data = identities[identity]
            camera_id = event.get("camera_id")
            similarity = float(event.get("similarity", 0.0))
            status = event.get("status", "matched")

            data["appearances"] += 1
            data["best_similarity"] = max(data["best_similarity"], similarity)
            if status == "needs_review":
                data["review_count"] += 1
            else:
                if camera_id is not None:
                    data["cameras"].add(camera_id)
            if event.get("is_new_identity"):
                data["new_identity_count"] += 1
            if status == "cross_camera_match":
                data["cross_camera_match_count"] += 1

            data["matches"].append({
                "camera_id": camera_id,
                "frame": event.get("frame"),
                "track_id": event.get("track_id"),
                "similarity": round(similarity, 4),
                "deep_similarity": round(float(event.get("deep_similarity", 0.0)), 4),
                "color_similarity": round(float(event.get("color_similarity", 0.0)), 4),
                "candidate_margin": round(float(event.get("candidate_margin", 0.0)), 4),
                "status": status,
                "is_new_identity": bool(event.get("is_new_identity", False)),
            })

    serializable = {}
    cross_camera = []
    for identity, data in sorted(identities.items()):
        cameras = sorted(c for c in data["cameras"] if c is not None)
        serializable[identity] = {
            "cameras": cameras,
            "appearances": data["appearances"],
            "best_similarity": round(data["best_similarity"], 4),
            "review_count": data["review_count"],
            "new_identity_count": data["new_identity_count"],
            "cross_camera_match_count": data["cross_camera_match_count"],
            "matches": data["matches"],
        }
        if len(cameras) > 1:
            cross_camera.append({"identity": identity, "cameras": cameras})

    summary = {
        "total_unique_identities": len(serializable),
        "cross_camera_identities": len(cross_camera),
        "cross_camera_matches": cross_camera,
        "identities": serializable,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PERSON RE-ID SUMMARY")
    print("=" * 60)
    print(f"Unique identities       : {summary['total_unique_identities']}")
    print(f"Cross-camera identities : {summary['cross_camera_identities']}")
    for identity, data in serializable.items():
        print(
            f"  {identity}: cameras={data['cameras']} "
            f"appearances={data['appearances']} "
            f"best_similarity={data['best_similarity']:.2f} "
            f"reviews={data['review_count']}"
        )
    print(f"Identity graph -> '{output_file}'")
    return summary


def _ordered_producer(video_sources, frame_queue, config, camera_metrics, log_file):
    """Produce cameras in order: Camera 1 -> Camera 2 -> Camera 3 -> ...

    This is intentional for the offline Re-ID demo. A shared in-memory gallery
    cannot make a correct decision if Camera 3 happens to arrive first and
    creates Person_001 before Camera 1/2 have established their identities.
    """
    for camera_id, source in enumerate(video_sources, start=1):
        print(f"[Producer] Starting Camera {camera_id} in ordered mode.")
        camera_producer(
            source,
            camera_id,
            frame_queue,
            config,
            camera_metrics,
            log_file,
        )
        log_event(log_file, {
            "level": "CAMERA_COMPLETE",
            "camera_id": camera_id,
        })


def run_pipeline(video_sources, config, object_classes):
    _reset_output(config)
    output = config["output"]
    output_dir = output["detections_dir"]
    log_file = output["log_file"]
    queue_maxsize = int(config["pipeline"].get("queue_maxsize", 64))

    # One worker is required because the Re-ID gallery lives in memory.
    if int(config["pipeline"].get("num_workers", 1)) != 1:
        print("[WARN] Shared global ReID gallery requires num_workers=1; forcing it.")

    frame_queue = queue.Queue(maxsize=queue_maxsize)
    producers_done = threading.Event()
    camera_metrics = {}

    producer = threading.Thread(
        target=_ordered_producer,
        args=(video_sources, frame_queue, config, camera_metrics, log_file),
        name="ordered-producer",
    )

    reid_cfg = config.get("reid", {})
    worker = threading.Thread(
        target=detection_worker,
        args=(
            1,
            frame_queue,
            producers_done,
            os.path.join("models", config["model"]["name"]),
            object_classes,
            float(config["detection"]["confidence_threshold"]),
            queue_maxsize,
            output_dir,
            log_file,
            reid_cfg,
            bool(output.get("save_all_frames", False)),
            int(output.get("selected_frames_per_identity", 1)),
            bool(output.get("save_annotated_videos", True)),
            output.get("annotated_video_dir", "results/annotated"),
        ),
        name="worker-1",
    )

    worker.start()
    producer.start()
    producer.join()
    producers_done.set()
    worker.join()

    print(
        f"\n[INFO] Pipeline complete.\n"
        f"Detections -> '{output_dir}/'\n"
        f"Events log -> '{log_file}'"
    )

    camera_data = build_camera_data(log_file)
    dashboard_path = output.get(
        "dashboard_file", "results/pipeline_monitoring_dashboard.png"
    )
    create_dashboard(camera_data, dashboard_path)

    identity_output = output.get(
        "identity_summary_file", "results/identity_summary.json"
    )
    build_identity_summary(log_file, identity_output)
