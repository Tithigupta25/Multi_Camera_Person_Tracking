import json
import statistics
from collections import defaultdict


def load_events(log_file):
    events = []

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return events

def build_summary(events):
   
    camera_data = defaultdict(lambda: {
            "frames_pushed": 0,
            "frames_dropped": 0,
            "drop_rate": 0,
            "queue_sizes": [],
            "queue_fullness": [],
            "processing_lags": [],
            "detections": 0})

    for event in events:
        camera_id = event.get("camera_id")

        if camera_id is None:
            continue

        data = camera_data[camera_id]

        if "frames_pushed" in event:
            data["frames_pushed"] = event["frames_pushed"]

        if "frames_dropped" in event:
            data["frames_dropped"] = event["frames_dropped"]

        if "frame_drop_rate" in event:
            data["drop_rate"] = event["frame_drop_rate"]

        if "queue_size" in event:
            data["queue_sizes"].append(event["queue_size"])

        if "queue_fullness" in event:
            data["queue_fullness"].append(event["queue_fullness"])

        if "processing_lag_sec" in event:
            data["processing_lags"].append(event["processing_lag_sec"])

        if event.get("level") == "EVENT":
            detections = event.get("detections", [])
            data["detections"] += len(detections)

    return camera_data

def print_summary(camera_data):
    print("\n" + "=" * 90)
    print("PIPELINE MONITORING SUMMARY")
    print("=" * 90)

    for camera_id in sorted(camera_data):
        data = camera_data[camera_id]
        max_queue = (max(data["queue_sizes"]) if data["queue_sizes"] else 0)
        max_fullness = (max(data["queue_fullness"]) if data["queue_fullness"] else 0)
        avg_lag = (statistics.mean(data["processing_lags"]) if data["processing_lags"] else 0)
        max_lag = (max(data["processing_lags"]) if data["processing_lags"] else 0)

        print(f"\nCamera {camera_id}")
        print(f"  Frames pushed : {data['frames_pushed']}")
        print(f"  Frames dropped: {data['frames_dropped']}")
        print(f"  Drop rate     : {data['drop_rate'] * 100:.2f}%")
        print(f"  Max queue     : {max_queue}")
        print(f"  Max fullness  : {max_fullness * 100:.1f}%")
        print(f"  Avg lag       : {avg_lag:.2f} sec")
        print(f"  Max lag       : {max_lag:.2f} sec")
        print(f"  Detections    : {data['detections']}")