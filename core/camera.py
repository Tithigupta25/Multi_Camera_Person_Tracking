import cv2
import time
from dataclasses import dataclass
import queue

from core.logger import log_event


@dataclass
class Frame:
    camera_id: int
    frame_number: int
    image: object
    timestamp: float


def camera_producer(
    video_path,
    camera_id,
    frame_queue,
    config,
    camera_metrics,
    log_file,
):
    """Read one source and push sampled frames into the bounded queue.

    For offline videos, blocking on a full queue is the default. This prevents
    the previous 70-80% frame-drop behaviour and, importantly, keeps all
    sampled frames from Camera 1 ahead of Camera 2/3 when the pipeline uses
    ordered production.
    """
    is_camera = isinstance(video_path, int)
    window_name = f"Camera {camera_id}"
    max_retries = int(config["pipeline"].get("max_retries", 3))
    frame_interval = max(1, int(config["detection"].get("frame_interval", 5)))
    max_frames = int(config["pipeline"].get("max_frames_per_camera", 500))
    drop_when_full = bool(config["pipeline"].get("drop_frames_when_full", False))

    attempt = 0
    cap = None
    while attempt < max_retries:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            break
        attempt += 1
        log_event(log_file, {
            "level": "WARN",
            "camera_id": camera_id,
            "msg": f"Failed to open stream, retry {attempt}/{max_retries}",
        })
        time.sleep(0.5)

    if cap is None or not cap.isOpened():
        log_event(log_file, {
            "level": "ERROR",
            "camera_id": camera_id,
            "msg": "Camera unreachable after max retries",
        })
        print(f"[Camera {camera_id}] ERROR: could not open {video_path}")
        return

    frame_number = 0
    pushed = 0
    dropped = 0
    total_sampled = 0

    while True:
        if not is_camera and frame_number >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            log_event(log_file, {
                "level": "INFO",
                "camera_id": camera_id,
                "msg": "Stream ended or frame read failed",
            })
            break

        if is_camera:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        now = time.time()
        camera_metrics[camera_id] = {
            "last_frame_received": now,
            "frame_number": frame_number,
        }

        if frame_number % frame_interval != 0:
            frame_number += 1
            continue

        total_sampled += 1
        item = Frame(camera_id, frame_number, frame, now)

        try:
            if drop_when_full:
                frame_queue.put_nowait(item)
            else:
                # Offline evaluation: do not silently lose sampled frames.
                frame_queue.put(item)
            pushed += 1
        except queue.Full:
            dropped += 1
            log_event(log_file, {
                "level": "METRIC",
                "camera_id": camera_id,
                "frame": frame_number,
                "frame_dropped": 1,
                "frames_dropped": dropped,
                "queue_size": frame_queue.qsize(),
            })

        frame_number += 1

    cap.release()
    if is_camera:
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    drop_rate = dropped / total_sampled if total_sampled > 0 else 0
    log_event(log_file, {
        "level": "METRIC",
        "camera_id": camera_id,
        "frames_pushed": pushed,
        "frames_dropped": dropped,
        "total_sampled": total_sampled,
        "frame_drop_rate": round(drop_rate, 3),
    })
    print(
        f"[Camera {camera_id}] Producer done | "
        f"Pushed: {pushed} | Dropped: {dropped} | "
        f"Drop rate: {drop_rate:.2%}"
    )
