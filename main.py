import os
import cv2

from core.config import load_config, load_classes
from core.pipeline import run_pipeline


def get_video_sources(config):
    input_config = config["input"]
    mode = input_config.get("mode", "auto")
    video_dir = input_config.get("video_dir", "videos")

    if mode == "camera":
        return input_config.get("camera_indices", [0])

    video_sources = []
    if os.path.isdir(video_dir):
        extensions = (".mp4", ".avi", ".mov", ".mkv")
        names = sorted(
            name for name in os.listdir(video_dir)
            if name.lower().endswith(extensions)
        )
        video_sources = [os.path.join(video_dir, name) for name in names]

    max_sources = int(input_config.get("max_video_sources", 0))
    if max_sources > 0:
        video_sources = video_sources[:max_sources]

    if video_sources:
        print(f"[INFO] Found {len(video_sources)} video source(s) in '{video_dir}'.")
        for camera_id, source in enumerate(video_sources, start=1):
            print(f"  Camera {camera_id}: {os.path.basename(source)}")
        return video_sources

    detected_cameras = []
    preferred_cameras = input_config.get("camera_indices", [])
    cameras_to_check = preferred_cameras if preferred_cameras else range(5)

    for index in cameras_to_check:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            detected_cameras.append(index)
        cap.release()

    if detected_cameras:
        selected = [i for i in preferred_cameras if i in detected_cameras]
        return selected or detected_cameras

    print("[ERROR] No cameras or video sources found.")
    return []


def main():
    config = load_config()
    object_classes = load_classes()
    video_sources = get_video_sources(config)
    if not video_sources:
        return
    run_pipeline(video_sources, config, object_classes)


if __name__ == "__main__":
    main()
