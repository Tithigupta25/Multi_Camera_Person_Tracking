import math


class SimpleTracker:
    """
    Lightweight centroid tracker used for the demo.

    Track IDs are local to each camera. Global identity is assigned separately
    by PersonReID.
    """

    def __init__(self, max_distance=80, max_missed=15):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections):
        # Age all existing tracks first.
        for track in self.tracks.values():
            track["missed"] += 1

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))

        candidates = []
        for track_id, track in self.tracks.items():
            tc = self._center(track["bbox"])
            for index, detection in enumerate(detections):
                if detection["label"] != track["label"]:
                    continue
                dc = self._center(detection["bbox"])
                candidates.append((self._distance(tc, dc), track_id, index))

        # Greedy nearest-neighbour assignment.
        for distance, track_id, index in sorted(candidates):
            if distance > self.max_distance:
                break
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue

            detection = detections[index]
            track = self.tracks[track_id]
            track.update({
                "bbox": detection["bbox"],
                "label": detection["label"],
                "confidence": detection["confidence"],
                "missed": 0,
            })
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        # Create tracks for new detections.
        for index in sorted(unmatched_detections):
            detection = detections[index]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "id": track_id,
                "bbox": detection["bbox"],
                "label": detection["label"],
                "confidence": detection["confidence"],
                "missed": 0,
            }

        # Remove stale tracks.
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]
        for track_id in stale:
            del self.tracks[track_id]

        # Return active tracks only.
        return [
            {
                "id": track["id"],
                "bbox": track["bbox"],
                "label": track["label"],
                "confidence": track["confidence"],
            }
            for track in self.tracks.values()
            if track["missed"] == 0
        ]
