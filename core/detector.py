import os
import queue
import time

import cv2
import numpy as np
from ultralytics import YOLO

from core.logger import log_event
from core.tracker import SimpleTracker
from core.reid import PersonReID


def _crop(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8), inter / (min(area_a, area_b) + 1e-8)


def _deduplicate_person_boxes(detections, iou_threshold=0.60, containment_threshold=0.90, min_size_ratio=0.55):
    """Remove duplicate person boxes produced by the detector.

    The original run contained almost-identical nested boxes around the same
    woman. Those duplicate boxes became separate global identities and then
    contaminated the Re-ID gallery. Keep the higher-confidence box when one
    person box is effectively contained in another.

    BUGFIX: containment alone (inter / smaller_area) cannot tell the
    difference between "two near-identical boxes on the same object" and
    "a small background person's box that happens to fall inside a large
    foreground person's box in a group scene". A background person's box is
    much SMALLER in area than the foreground person's box, whereas true
    duplicate boxes on the same detection are close to the same size. Only
    treat high containment as a duplicate when the two boxes are also
    similar in size (area ratio >= min_size_ratio) -- this keeps the
    original fix for real duplicate/nested boxes while no longer deleting
    genuinely different, smaller background people.
    """
    persons = [d for d in detections if d["label"].lower() == "person"]
    others = [d for d in detections if d["label"].lower() != "person"]
    persons.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for candidate in persons:
        duplicate = False
        for existing in kept:
            iou, containment = _iou(candidate["bbox"], existing["bbox"])
            area_a = _box_area(candidate["bbox"])
            area_b = _box_area(existing["bbox"])
            size_ratio = min(area_a, area_b) / (max(area_a, area_b) + 1e-8)
            same_size_enough = size_ratio >= min_size_ratio
            if same_size_enough and (containment >= containment_threshold
                                      or (iou >= iou_threshold and containment >= 0.80)):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept + others


def _box_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _mean_feature(features):
    if not features:
        return None
    value = np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)
    return value / (np.linalg.norm(value) + 1e-8)


def detection_worker(
    worker_id,
    frame_queue,
    producers_done_event,
    model_path,
    object_classes,
    confidence_threshold,
    queue_maxsize,
    output_dir,
    log_file,
    reid_config=None,
    save_all_frames=False,
    selected_frames_per_identity=1,
    save_annotated_videos=True,
    annotated_video_dir="results/annotated",
):
    print(f"[Worker {worker_id}] Started.")

    model = YOLO(model_path)
    trackers = {}
    reid_config = reid_config or {}
    reid = PersonReID(
        similarity_threshold=reid_config.get("similarity_threshold", 0.70),
        review_margin=reid_config.get("review_margin", 0.05),
        deep_weight=reid_config.get("deep_weight", 0.60),
        color_weight=reid_config.get("color_weight", 0.40),
        min_deep_similarity=reid_config.get("min_deep_similarity", 0.65),
        min_color_similarity=reid_config.get("min_color_similarity", 0.55),
        min_candidate_margin=reid_config.get("min_candidate_margin", 0.03),
        update_alpha=reid_config.get("update_alpha", 0.10),
        max_prototypes=reid_config.get("max_prototypes", 16),
        allow_same_camera_reid=reid_config.get("allow_same_camera_reid", False),
        camera_thresholds=reid_config.get("camera_thresholds", {}),
        camera_deep_thresholds=reid_config.get("camera_deep_thresholds", {}),
        audit_log_path=reid_config.get("audit_log_file", "results/identity_audit.jsonl"),
    )

    track_identity_cache = {}
    reid_observations = {}
    saved_identity_camera = set()
    last_printed_decision = {}
    video_writers = {}
    camera_counts = {}

    os.makedirs(output_dir, exist_ok=True)
    if save_annotated_videos:
        os.makedirs(annotated_video_dir, exist_ok=True)

    observation_count = max(1, int(reid_config.get("observation_count", 4)))
    observation_interval = max(1, int(reid_config.get("observation_interval", 5)))
    max_observations = max(observation_count, int(reid_config.get("max_observations", 8)))
    duplicate_iou = float(reid_config.get("duplicate_iou_threshold", 0.60))
    duplicate_containment = float(reid_config.get("duplicate_containment_threshold", 0.90))

    processed = 0
    total_detections = 0
    total_tracked_objects = 0
    total_person_decisions = 0
    total_cross_camera = 0
    total_review = 0
    start = time.time()

    def writer_for(camera_id, image):
        if not save_annotated_videos or camera_id in video_writers:
            return
        h, w = image.shape[:2]
        path = os.path.join(annotated_video_dir, f"camera{camera_id}_annotated.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        if writer.isOpened():
            video_writers[camera_id] = writer
        else:
            writer.release()

    while True:
        try:
            item = frame_queue.get(timeout=2)
        except queue.Empty:
            if producers_done_event.is_set() and frame_queue.empty():
                break
            continue

        if item is None:
            frame_queue.task_done()
            break

        try:
            camera_id = item.camera_id
            writer_for(camera_id, item.image)
            if camera_id not in trackers:
                trackers[camera_id] = SimpleTracker(max_distance=90, max_missed=12)
                camera_counts[camera_id] = {"frames": 0, "detections": 0, "tracks": set()}
                print(f"[Worker {worker_id}] Camera {camera_id} started.")

            tracker = trackers[camera_id]
            result = model(item.image, verbose=False)[0]
            detections = []
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if object_classes and label not in object_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"label": label, "confidence": round(conf, 3), "bbox": [x1, y1, x2, y2]})

            before = len(detections)
            detections = _deduplicate_person_boxes(detections, duplicate_iou, duplicate_containment)
            duplicates_removed = before - len(detections)
            if duplicates_removed:
                log_event(log_file, {"level": "DETECTION_CLEANUP", "camera_id": camera_id,
                                     "frame": item.frame_number, "duplicates_removed": duplicates_removed})

            tracked_objects = tracker.update(detections)
            total_detections += len(detections)
            total_tracked_objects += len(tracked_objects)
            camera_counts[camera_id]["frames"] += 1
            camera_counts[camera_id]["detections"] += len(detections)
            camera_counts[camera_id]["tracks"].update(t["id"] for t in tracked_objects)

            # Only identities attached to tracks that are ACTIVE in this frame
            # are excluded. Previously every identity ever seen by this camera
            # was excluded, which made same-camera track recovery impossible.
            active_track_ids = {int(t["id"]) for t in tracked_objects if t.get("label", "").lower() == "person"}
            active_identities = {
                track_identity_cache[(camera_id, tid)]["identity"]
                for tid in active_track_ids
                if (camera_id, tid) in track_identity_cache
                and track_identity_cache[(camera_id, tid)].get("identity")
            }

            track_events = []
            identities_assigned_this_frame = []

            for track in tracked_objects:
                track_id = track["id"]
                label = track["label"]
                confidence = float(track["confidence"])
                bbox = track["bbox"]
                identity = None
                similarity = 0.0
                status = "not_applicable"
                is_new_identity = False
                deep_similarity = color_similarity = candidate_margin = 0.0
                candidate = None

                if label.lower() == "person":
                    cache_key = (camera_id, track_id)
                    if cache_key in track_identity_cache:
                        cached = track_identity_cache[cache_key]
                        identity = cached.get("identity")
                        similarity = cached.get("similarity", 0.0)
                        status = cached.get("status", "matched")
                        is_new_identity = cached.get("is_new", False)
                        deep_similarity = cached.get("deep_similarity", 0.0)
                        color_similarity = cached.get("color_similarity", 0.0)
                        candidate_margin = cached.get("candidate_margin", 0.0)
                    else:
                        state = reid_observations.setdefault(cache_key, {"deep": [], "color": [], "last_frame": -1})
                        if state["last_frame"] < 0 or item.frame_number - state["last_frame"] >= observation_interval:
                            crop = _crop(item.image, bbox)
                            deep, color = reid.extract_features(crop)
                            if deep is not None:
                                state["deep"].append(deep)
                            if color is not None:
                                state["color"].append(color)
                            state["last_frame"] = item.frame_number

                        sample_count = len(state["deep"])
                        ready = sample_count >= observation_count
                        should_retry = sample_count <= max_observations and (ready or sample_count == max_observations)
                        if should_retry:
                            avg_deep = _mean_feature(state["deep"])
                            avg_color = _mean_feature(state["color"])
                            result_reid = reid.identify(
                                camera_id=camera_id,
                                exclude_identities=active_identities,
                                deep=avg_deep,
                                color=avg_color,
                            )
                            identity = result_reid.get("identity")
                            similarity = float(result_reid.get("similarity", 0.0))
                            status = result_reid.get("status", "needs_review")
                            is_new_identity = bool(result_reid.get("is_new", False))
                            deep_similarity = float(result_reid.get("deep_similarity", 0.0))
                            color_similarity = float(result_reid.get("color_similarity", 0.0))
                            candidate_margin = float(result_reid.get("candidate_margin", 0.0))
                            candidate = result_reid.get("candidate")

                            # NEVER turn an ambiguous match into a new identity.
                            # That was the source of the duplicate Person_XXX IDs
                            # seen when the same person appeared in another camera.
                            # Keep the track in review until a later observation
                            # becomes decisive.

                            final_decision = status != "needs_review"
                            if final_decision:
                                track_identity_cache[cache_key] = {
                                    "identity": identity,
                                    "similarity": similarity,
                                    "status": status,
                                    "is_new": is_new_identity,
                                    "deep_similarity": deep_similarity,
                                    "color_similarity": color_similarity,
                                    "candidate_margin": candidate_margin,
                                    "candidate": candidate,
                                }
                                reid_observations.pop(cache_key, None)
                            total_person_decisions += 1

                            if status == "cross_camera_match":
                                total_cross_camera += 1
                            elif status == "needs_review":
                                total_review += 1

                            if identity:
                                active_identities.add(identity)
                                identities_assigned_this_frame.append(identity)

                            log_event(log_file, {
                                "level": "IDENTITY_MATCH",
                                "worker": worker_id,
                                "camera_id": camera_id,
                                "frame": item.frame_number,
                                "track_id": track_id,
                                "identity": identity,
                                "candidate": candidate,
                                "similarity": round(similarity, 4),
                                "deep_similarity": round(deep_similarity, 4),
                                "color_similarity": round(color_similarity, 4),
                                "candidate_margin": round(candidate_margin, 4),
                                "status": status,
                                "is_new_identity": is_new_identity,
                            })
                        else:
                            status = "reid_pending"

                # Draw
                cv2.rectangle(item.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                if identity:
                    text = f"T{track_id} | {identity} | {similarity:.2f} | {status}"
                elif status == "needs_review":
                    text = f"T{track_id} | needs_review | {similarity:.2f}"
                else:
                    text = f"T{track_id} | {label} {confidence:.2f} | {status}"
                cv2.putText(item.image, text, (bbox[0], max(bbox[1] - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2)

                track_events.append({
                    "track_id": track_id, "label": label, "confidence": round(confidence, 3),
                    "bbox": bbox, "identity": identity, "similarity": round(similarity, 4),
                    "deep_similarity": round(deep_similarity, 4), "color_similarity": round(color_similarity, 4),
                    "candidate_margin": round(candidate_margin, 4), "candidate": candidate,
                    "identity_status": status, "is_new_identity": is_new_identity,
                })

            # Save one representative image per confirmed identity/camera.
            if save_all_frames and tracked_objects:
                cv2.imwrite(os.path.join(output_dir, f"cam{camera_id}_frame{item.frame_number:05d}.jpg"), item.image)
            else:
                for identity in identities_assigned_this_frame:
                    key = (camera_id, identity)
                    if key in saved_identity_camera or selected_frames_per_identity <= 0:
                        continue
                    out_path = os.path.join(output_dir, f"cam{camera_id}_{identity}_frame{item.frame_number:05d}.jpg")
                    if cv2.imwrite(out_path, item.image):
                        saved_identity_camera.add(key)

            if camera_id in video_writers:
                video_writers[camera_id].write(item.image)

            # Only print when something changed for a track (not every sampled
            # frame) to avoid flooding the terminal. Person-only by design:
            # even if classes.txt or the model ever surfaces a non-person
            # label (e.g. "chair"), it is filtered out upstream by
            # object_classes and never reaches this print at all.
            for t in track_events:
                if t["label"].lower() != "person":
                    continue
                key = (camera_id, t["track_id"])
                decision = (t["identity"], t["identity_status"], round(t["similarity"], 3))
                if decision != last_printed_decision.get(key):
                    last_printed_decision[key] = decision
                    if t["identity"] or t["identity_status"] in {"needs_review", "new_identity"}:
                        print(f"[ReID] Cam{camera_id} T{t['track_id']} -> "
                              f"{t['identity'] or t['identity_status']} "
                              f"score={t['similarity']:.3f}")

            log_event(log_file, {
                "level": "EVENT", "camera_id": camera_id, "frame": item.frame_number,
                "worker": worker_id, "detections": detections, "tracks": track_events,
            })
            processed += 1

        except Exception as exc:
            log_event(log_file, {"level": "ERROR", "worker": worker_id,
                                 "camera_id": getattr(item, "camera_id", None),
                                 "frame": getattr(item, "frame_number", None), "msg": str(exc)})
            print(f"[Worker {worker_id}] ERROR: {exc}")
        finally:
            frame_queue.task_done()

    for writer in video_writers.values():
        writer.release()

    elapsed = time.time() - start
    fps = processed / elapsed if elapsed > 0 else 0
    print("\n[Worker] Done.")
    print(f"Processed sampled frames : {processed}")
    print(f"Detections              : {total_detections}")
    print(f"Tracked objects         : {total_tracked_objects}")
    print(f"Identity decisions      : {total_person_decisions}")
    print(f"Cross-camera matches    : {total_cross_camera}")
    print(f"Needs review            : {total_review}")
    print(f"Known global identities : {len(reid.records)}")
    print(f"Processing speed        : {fps:.2f} FPS")
    for cam_id in sorted(camera_counts):
        c = camera_counts[cam_id]
        print(f"Camera {cam_id}: frames={c['frames']} detections={c['detections']} tracks={len(c['tracks'])}")
