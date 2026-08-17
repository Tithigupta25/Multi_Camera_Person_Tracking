import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class IdentityRecord:
    identity: str
    prototypes: list = field(default_factory=list)
    color_prototypes: list = field(default_factory=list)
    cameras_seen: set = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    sightings: int = 0
    consent_status: str = "pending"
    consent_scope: str = "security_only"


class PersonReID:
    """Conservative shared gallery for offline multi-camera person Re-ID.

    Local tracker IDs (T1/T2/...) never become global identities directly.
    A global identity is created only after a stable track has accumulated
    several observations. Cross-camera matches use configurable camera-specific
    thresholds because the supplied videos have very different viewpoints.

    The bundled ResNet18 is an ImageNet backbone, not a dedicated person-ReID
    model. This is therefore a demo/assessment pipeline, not a production
    biometric identification system.
    """

    def __init__(
        self,
        similarity_threshold=0.70,
        review_margin=0.04,
        deep_weight=0.60,
        color_weight=0.40,
        min_deep_similarity=0.65,
        min_color_similarity=0.55,
        min_candidate_margin=0.02,
        update_alpha=0.10,
        max_prototypes=16,
        allow_same_camera_reid=False,
        camera_thresholds=None,
        camera_deep_thresholds=None,
        audit_log_path="results/identity_audit.jsonl",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = float(similarity_threshold)
        self.review_margin = float(review_margin)
        self.deep_weight = float(deep_weight)
        self.color_weight = float(color_weight)
        self.min_deep = float(min_deep_similarity)
        self.min_color = float(min_color_similarity)
        self.min_margin = float(min_candidate_margin)
        self.update_alpha = float(update_alpha)
        self.max_prototypes = int(max_prototypes)
        self.allow_same_camera_reid = bool(allow_same_camera_reid)
        self.camera_thresholds = {int(k): float(v) for k, v in (camera_thresholds or {}).items()}
        self.camera_deep_thresholds = {
            int(k): float(v) for k, v in (camera_deep_thresholds or {}).items()
        }

        # Be tolerant of hand-edited config files. The two weights are normalized
        # instead of crashing the whole worker when they do not add to exactly 1.0.
        weight_sum = self.deep_weight + self.color_weight
        if weight_sum <= 0.0:
            self.deep_weight, self.color_weight = 0.60, 0.40
        elif abs(weight_sum - 1.0) > 1e-6:
            self.deep_weight /= weight_sum
            self.color_weight /= weight_sum

        self.records = {}
        self.next_identity = 1
        self.audit_log_path = audit_log_path
        os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)

        print(f"[ReID] Device={self.device} | thresholds={self.camera_thresholds or {'default': self.threshold}}")

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval().to(self.device)
        self.transform = weights.transforms()

    def _audit(self, event):
        payload = dict(event)
        payload["ts"] = time.time()
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size == 0 or a.size != b.size:
            return 0.0
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    @staticmethod
    def _prepare_crop(image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 40 or w < 20:
            return None
        x1, x2 = int(w * 0.04), int(w * 0.96)
        y1, y2 = int(h * 0.02), int(h * 0.98)
        crop = image[y1:y2, x1:x2]
        return crop if crop.size else image

    @classmethod
    def _color_feature(cls, image):
        image = cls._prepare_crop(image)
        if image is None:
            return None
        crop = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        parts = []
        h = crop.shape[0]
        for r in range(4):
            y1, y2 = r * h // 4, (r + 1) * h // 4
            band_hsv, band_lab = hsv[y1:y2], lab[y1:y2]
            hist = cv2.calcHist([band_hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten().astype(np.float32)
            hist /= np.linalg.norm(hist) + 1e-8
            mean_lab = band_lab.mean(axis=(0, 1)).astype(np.float32) / 255.0
            std_lab = band_lab.std(axis=(0, 1)).astype(np.float32) / 255.0
            parts.extend([hist, mean_lab, std_lab])
        return cls._normalize(np.concatenate(parts).astype(np.float32))

    def extract_embedding(self, image):
        image = self._prepare_crop(image)
        if image is None:
            return None
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                deep = self.model(tensor).flatten()
            deep = deep / (torch.norm(deep) + 1e-8)
            return deep.cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def extract_features(self, image):
        return self.extract_embedding(image), self._color_feature(image)

    def _new_identity(self, deep, color, camera_id):
        identity = f"Person_{self.next_identity:03d}"
        self.next_identity += 1
        now = time.time()
        self.records[identity] = IdentityRecord(
            identity=identity,
            prototypes=[deep.copy()] if deep is not None else [],
            color_prototypes=[color.copy()] if color is not None else [],
            cameras_seen={camera_id} if camera_id is not None else set(),
            first_seen=now,
            last_seen=now,
            sightings=1,
        )
        return identity

    def add_observation(self, identity, deep, color, camera_id=None):
        """Update an already assigned identity with a stable track observation."""
        if identity not in self.records:
            return
        rec = self.records[identity]
        if deep is not None:
            rec.prototypes.append(np.asarray(deep, dtype=np.float32).copy())
            if len(rec.prototypes) > self.max_prototypes:
                rec.prototypes.pop(0)
        if color is not None:
            rec.color_prototypes.append(np.asarray(color, dtype=np.float32).copy())
            if len(rec.color_prototypes) > self.max_prototypes:
                rec.color_prototypes.pop(0)
        if camera_id is not None:
            rec.cameras_seen.add(camera_id)
        rec.last_seen = time.time()
        rec.sightings += 1

    def _candidate_score(self, deep, color, record):
        # Top-k mean is more stable than max: a single accidental close frame
        # should not be enough to merge two people.
        deep_scores = sorted((self._cosine(deep, p) for p in record.prototypes), reverse=True)
        color_scores = sorted((self._cosine(color, p) for p in record.color_prototypes), reverse=True)
        k = min(3, len(deep_scores))
        deep_score = float(np.mean(deep_scores[:k])) if k else 0.0
        k = min(3, len(color_scores))
        color_score = float(np.mean(color_scores[:k])) if k else 0.0
        combined = self.deep_weight * deep_score + self.color_weight * color_score
        return combined, deep_score, color_score

    def create_identity_from_features(self, deep, color, camera_id):
        identity = self._new_identity(deep, color, camera_id)
        self._audit({"event": "identity_created", "identity": identity, "camera_id": camera_id,
                     "status": "new_identity", "reason": "review_resolved_as_new"})
        return identity

    def _threshold_for(self, camera_id):
        return self.camera_thresholds.get(camera_id, self.threshold)

    def _deep_threshold_for(self, camera_id):
        return self.camera_deep_thresholds.get(camera_id, self.min_deep)

    def identify(self, camera_id=None, exclude_identities=None, deep=None, color=None, image=None):
        if deep is None and color is None and image is not None:
            deep, color = self.extract_features(image)
        if deep is None:
            return self._result(None, 0, 0, 0, 0, "embedding_failed", False)

        excluded = set(exclude_identities or [])
        candidates = []
        for identity, record in self.records.items():
            if identity in excluded:
                continue
            if not self.allow_same_camera_reid and camera_id in record.cameras_seen:
                continue
            score, deep_score, color_score = self._candidate_score(deep, color, record)
            candidates.append((score, identity, deep_score, color_score))

        candidates.sort(reverse=True, key=lambda x: x[0])
        if not candidates:
            identity = self._new_identity(deep, color, camera_id)
            self._audit({"event": "identity_created", "identity": identity, "camera_id": camera_id,
                         "status": "new_identity", "reason": "no_eligible_candidate"})
            return self._result(identity, 0, 0, 0, 0, "new_identity", True)

        best_score, best_id, best_deep, best_color = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = best_score - second_score
        threshold = self._threshold_for(camera_id)
        deep_threshold = self._deep_threshold_for(camera_id)

        strong = (
            best_score >= threshold
            and best_deep >= deep_threshold
            and best_color >= self.min_color
            and margin >= self.min_margin
        )

        if strong:
            previous_cameras = set(self.records[best_id].cameras_seen)
            self.add_observation(best_id, deep, color, camera_id)
            status = "cross_camera_match" if camera_id not in previous_cameras else "matched"
            self._audit({
                "event": "identity_match", "identity": best_id, "camera_id": camera_id,
                "similarity": round(best_score, 4), "deep_similarity": round(best_deep, 4),
                "color_similarity": round(best_color, 4), "candidate_margin": round(margin, 4),
                "status": status,
            })
            return self._result(best_id, best_score, best_deep, best_color, margin, status, False)

        # Borderline results are returned without an identity. The old version
        # displayed the best candidate as if it were a confirmed identity,
        # which is exactly how the false Cam3 -> Person_005 match was shown.
        borderline = (
            best_score >= threshold - self.review_margin
            or best_deep >= deep_threshold - 0.04
        )
        if borderline:
            self._audit({
                "event": "needs_review", "candidate": best_id, "camera_id": camera_id,
                "similarity": round(best_score, 4), "deep_similarity": round(best_deep, 4),
                "color_similarity": round(best_color, 4), "candidate_margin": round(margin, 4),
                "threshold": threshold,
            })
            return self._result(None, best_score, best_deep, best_color, margin, "needs_review", False,
                                candidate=best_id)

        identity = self._new_identity(deep, color, camera_id)
        self._audit({
            "event": "identity_created", "identity": identity, "camera_id": camera_id,
            "similarity": round(best_score, 4), "best_candidate": best_id,
            "status": "new_identity", "reason": "below_match_threshold",
        })
        return self._result(identity, best_score, best_deep, best_color, margin, "new_identity", True,
                            candidate=best_id)

    @staticmethod
    def _result(identity, similarity, deep, color, margin, status, is_new, candidate=None):
        return {
            "identity": identity,
            "similarity": float(similarity),
            "deep_similarity": float(deep),
            "color_similarity": float(color),
            "candidate_margin": float(margin),
            "status": status,
            "is_new": bool(is_new),
            "candidate": candidate,
        }

    def summary(self):
        return {
            identity: {
                "cameras": sorted(rec.cameras_seen),
                "appearances": rec.sightings,
                "prototype_count": len(rec.prototypes),
                "consent_status": rec.consent_status,
                "consent_scope": rec.consent_scope,
            }
            for identity, rec in sorted(self.records.items())
        }
