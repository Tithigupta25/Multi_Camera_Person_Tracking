import time


class ActivityAnalyzer:

    def __init__(self, config):

        activity_config = config["activity"]

        self.loitering_seconds = activity_config["loitering_seconds"]
        self.restricted_confirm_frames = activity_config["restricted_confirm_frames"]

        self.crowd_min_people = activity_config["crowd_min_people"]
        self.crowd_confirm_seconds = activity_config["crowd_confirm_seconds"]

        self.fall_confirm_frames = activity_config["fall_confirm_frames"]
        self.fall_aspect_ratio = activity_config["fall_aspect_ratio"]

        self.running_confirm_frames = activity_config["running_confirm_frames"]
        self.running_distance_threshold = activity_config["running_distance_threshold"]

        self.fighting_confirm_frames = activity_config["fighting_confirm_frames"]
        self.fighting_distance_threshold = activity_config["fighting_distance_threshold"]

        # Restricted / Loitering
        self.restricted_start = {}
        self.restricted_frames = {}
        self.restricted_confirmed = {}
        self.loitering_confirmed = {}

        # Fall
        self.previous_bbox = {}
        self.fall_frames = {}
        self.fall_confirmed = {}

        # Running
        self.previous_centers = {}
        self.running_frames = {}
        self.running_confirmed = {}

        # Crowd
        self.crowd_start = None
        self.crowd_confirmed = False

        # Fighting
        self.fighting_frames = 0
        self.fighting_confirmed = False

    def update(self, tracked_people, zone_manager):

        events = []
        current_time = time.time()
        current_ids = set()

        for person in tracked_people:

            person_id = person["id"]
            bbox = person["bbox"]

            current_ids.add(person_id)

            x1, y1, x2, y2 = bbox

            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)

            # ========================================================
            # FALL DETECTION
            # ========================================================

            previous = self.previous_bbox.get(person_id)

            current_ratio = width / height

            fall_candidate = False

            if previous is not None:

                prev_x1, prev_y1, prev_x2, prev_y2 = previous

                prev_width = max(prev_x2 - prev_x1, 1)
                prev_height = max(prev_y2 - prev_y1, 1)

                prev_ratio = prev_width / prev_height

                # Person changes from vertical to horizontal
                became_horizontal = (
                    prev_ratio < self.fall_aspect_ratio
                    and
                    current_ratio >= self.fall_aspect_ratio
                )

                # Current person is horizontal
                is_horizontal = (
                    current_ratio >= self.fall_aspect_ratio
                )

                # Person moved downward
                vertical_drop = (
                    y2 - prev_y2 > prev_height * 0.20
                )

                # ----------------------------------------------------
                # Fall candidate
                # ----------------------------------------------------
                #
                # 1. Vertical -> horizontal
                # 2. Horizontal + downward movement
                # 3. Once fall starts, keep confirming while horizontal
                #

                if became_horizontal:

                    fall_candidate = True

                elif is_horizontal and vertical_drop:

                    fall_candidate = True

                elif (
                    is_horizontal
                    and
                    self.fall_frames.get(person_id, 0) > 0
                ):

                    fall_candidate = True

            # Update fall confirmation frames
            if fall_candidate:

                self.fall_frames[person_id] = (
                    self.fall_frames.get(person_id, 0) + 1
                )

            else:

                self.fall_frames[person_id] = 0

            # --------------------------------------------------------
            # IMPORTANT:
            # If a fall candidate is detected, do NOT allow running
            # during the fall confirmation period.
            # --------------------------------------------------------

            if fall_candidate:

                self.running_frames[person_id] = 0
                self.running_confirmed[person_id] = False

            # Confirm fall
            if (
                self.fall_frames.get(person_id, 0)
                >= self.fall_confirm_frames
                and
                not self.fall_confirmed.get(person_id, False)
            ):

                events.append({
                    "type": "fall",
                    "person_id": person_id,
                    "duration": 0
                })

                self.fall_confirmed[person_id] = True

                # A fallen person should never be considered running
                self.running_frames[person_id] = 0
                self.running_confirmed[person_id] = False

            self.previous_bbox[person_id] = bbox

            # ========================================================
            # RUNNING DETECTION
            # ========================================================

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            current_center = (center_x, center_y)
            previous_center = self.previous_centers.get(person_id)

            moving_fast = False

            # Current person shape
            current_ratio = width / height

            # Do NOT classify person as running when:
            #
            # 1. Fall already confirmed
            # 2. Fall is currently being confirmed
            # 3. Person is already horizontal
            #
            # This prevents fall -> running false detection.

            fall_in_progress = (
                self.fall_frames.get(person_id, 0) > 0
            )

            is_horizontal = (
                current_ratio >= self.fall_aspect_ratio
            )

            if (
                not self.fall_confirmed.get(person_id, False)
                and
                not fall_in_progress
                and
                not is_horizontal
            ):

                if previous_center is not None:

                    previous_x, previous_y = previous_center

                    distance = (
                        (center_x - previous_x) ** 2
                        +
                        (center_y - previous_y) ** 2
                    ) ** 0.5

                    if distance >= self.running_distance_threshold:

                        moving_fast = True

                if moving_fast:

                    self.running_frames[person_id] = (
                        self.running_frames.get(person_id, 0) + 1
                    )

                else:

                    self.running_frames[person_id] = 0

                # Confirm running
                if (
                    self.running_frames.get(person_id, 0)
                    >= self.running_confirm_frames
                    and
                    not self.running_confirmed.get(person_id, False)
                ):

                    events.append({
                        "type": "running",
                        "person_id": person_id
                    })

                    self.running_confirmed[person_id] = True

            else:

                # Disable running during/after fall
                self.running_frames[person_id] = 0
                self.running_confirmed[person_id] = False

            self.previous_centers[person_id] = current_center

            # ========================================================
            # RESTRICTED AREA + LOITERING
            # ========================================================

            zone = zone_manager.get_zone(bbox)

            if zone == "restricted":

                if person_id not in self.restricted_start:

                    self.restricted_start[person_id] = current_time
                    self.restricted_frames[person_id] = 0
                    self.restricted_confirmed[person_id] = False
                    self.loitering_confirmed[person_id] = False

                self.restricted_frames[person_id] += 1

                elapsed = (
                    current_time
                    -
                    self.restricted_start[person_id]
                )

                # Restricted area confirmation
                if (
                    self.restricted_frames[person_id]
                    >= self.restricted_confirm_frames
                    and
                    not self.restricted_confirmed.get(
                        person_id,
                        False
                    )
                ):

                    events.append({
                        "type": "restricted_area",
                        "person_id": person_id,
                        "zone": zone,
                        "duration": round(elapsed, 2)
                    })

                    self.restricted_confirmed[person_id] = True

                # Loitering confirmation
                if (
                    elapsed >= self.loitering_seconds
                    and
                    not self.loitering_confirmed.get(
                        person_id,
                        False
                    )
                ):

                    events.append({
                        "type": "loitering",
                        "person_id": person_id,
                        "zone": zone,
                        "duration": round(elapsed, 2)
                    })

                    self.loitering_confirmed[person_id] = True

            else:

                self.restricted_start.pop(person_id, None)
                self.restricted_frames.pop(person_id, None)
                self.restricted_confirmed.pop(person_id, None)
                self.loitering_confirmed.pop(person_id, None)

        # ============================================================
        # CLEANUP DISAPPEARED PEOPLE
        # ============================================================

        tracked_state_ids = (
            set(self.restricted_start.keys())
            |
            set(self.fall_frames.keys())
            |
            set(self.previous_bbox.keys())
            |
            set(self.previous_centers.keys())
        )

        disappeared_ids = tracked_state_ids - current_ids

        for person_id in disappeared_ids:

            self.restricted_start.pop(person_id, None)
            self.restricted_frames.pop(person_id, None)
            self.restricted_confirmed.pop(person_id, None)
            self.loitering_confirmed.pop(person_id, None)

            self.fall_frames.pop(person_id, None)
            self.fall_confirmed.pop(person_id, None)
            self.previous_bbox.pop(person_id, None)

            self.previous_centers.pop(person_id, None)
            self.running_frames.pop(person_id, None)
            self.running_confirmed.pop(person_id, None)

        # ============================================================
        # CROWD DETECTION
        # ============================================================

        people_count = len(tracked_people)

        if people_count >= self.crowd_min_people:

            if self.crowd_start is None:

                self.crowd_start = current_time
                self.crowd_confirmed = False

            crowd_duration = (
                current_time
                -
                self.crowd_start
            )

            if (
                crowd_duration >= self.crowd_confirm_seconds
                and
                not self.crowd_confirmed
            ):

                events.append({
                    "type": "crowd",
                    "people_count": people_count,
                    "duration": round(crowd_duration, 2)
                })

                self.crowd_confirmed = True

        else:

            self.crowd_start = None
            self.crowd_confirmed = False

        # ============================================================
        # FIGHTING DETECTION
        # ============================================================

        fighting_detected = False

        for i in range(len(tracked_people)):

            for j in range(i + 1, len(tracked_people)):

                bbox1 = tracked_people[i]["bbox"]
                bbox2 = tracked_people[j]["bbox"]

                ax1, ay1, ax2, ay2 = bbox1
                bx1, by1, bx2, by2 = bbox2

                center1 = (
                    (ax1 + ax2) / 2,
                    (ay1 + ay2) / 2
                )

                center2 = (
                    (bx1 + bx2) / 2,
                    (by1 + by2) / 2
                )

                distance = (
                    (center1[0] - center2[0]) ** 2
                    +
                    (center1[1] - center2[1]) ** 2
                ) ** 0.5

                height1 = max(ay2 - ay1, 1)
                height2 = max(by2 - by1, 1)

                average_height = (
                    height1 + height2
                ) / 2

                if (
                    distance
                    <
                    average_height
                    * self.fighting_distance_threshold
                ):

                    fighting_detected = True
                    break

            if fighting_detected:
                break

        # Fighting confirmation
        if fighting_detected:

            self.fighting_frames += 1

        else:

            self.fighting_frames = 0
            self.fighting_confirmed = False

        if (
            self.fighting_frames
            >= self.fighting_confirm_frames
            and
            not self.fighting_confirmed
        ):

            events.append({
                "type": "fighting",
                "people_count": people_count
            })

            self.fighting_confirmed = True

        return events