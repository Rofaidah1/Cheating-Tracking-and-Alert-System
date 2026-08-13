from pathlib import Path
from datetime import datetime
import math
import cv2
import numpy as np
from ultralytics import YOLO

from database import insert_alert
from notifier import send_cheating_notification

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOT_DIR = BASE_DIR / "alerts" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = 0
PHONE_CLASS = 67

START_CONFIRM_FRAMES = 3
END_GRACE_FRAMES = 90
INCIDENT_COOLDOWN_SECONDS = 20










WARMUP_FRAMES = 45
SLOT_MEMORY_FRAMES = 900
MAX_SLOT_DISTANCE = 4.0
MAX_ASSIGNMENT_DISTANCE = 5.0


class CheatingDetector:
    def __init__(self, model_name=MODEL_NAME):
        self.model = YOLO(model_name)

        self.frame_number = 0



        self.slots = {}
        self.next_student_id = 1


        self.events = {}

    @staticmethod
    def center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
        ab = max(0.0, bx2-bx1) * max(0.0, by2-by1)
        union = aa + ab - inter
        return inter / union if union else 0.0

    @staticmethod
    def appearance_signature(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, x2 = max(0, min(w-1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h-1, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (48, 96))
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 12],
                            [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    @staticmethod
    def appearance_similarity(a, b):
        if a is None or b is None:
            return 0.0
        d = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - float(d))

    def _slot_scale(self, box):
        return max(1.0, box[2]-box[0], box[3]-box[1])

    def _assignment_cost(self, person, slot):
        pc = self.center(person["box"])
        sc = slot["center"]
        scale = self._slot_scale(person["box"])
        d = self.distance(pc, sc) / scale
        if d > MAX_ASSIGNMENT_DISTANCE:
            return None

        iou = self.iou(person["box"], slot["box"])
        app = self.appearance_similarity(
            person.get("signature"), slot.get("signature")
        )

        cost = (
            0.65 * min(1.0, d / MAX_ASSIGNMENT_DISTANCE)
            + 0.20 * (1.0 - iou)
            + 0.15 * (1.0 - app)
        )
        return cost

    def update_slots(self, persons, frame):
        """
        Application-level identity.

        The first stable observations become seat slots. Afterwards,
        detections are matched one-to-one to those slots. We NEVER use
        the YOLO/BoT-SORT raw ID as the student ID shown to the user.
        """
        for p in persons:
            p["signature"] = self.appearance_signature(frame, p["box"])


        if self.frame_number <= WARMUP_FRAMES:
            for p in persons:
                c = self.center(p["box"])

                nearest = None
                nearest_d = 1e9
                for sid, slot in self.slots.items():
                    d = self.distance(c, slot["center"]) / self._slot_scale(p["box"])
                    if d < nearest_d:
                        nearest_d, nearest = d, sid

                if nearest is not None and nearest_d < 1.2:
                    slot = self.slots[nearest]
                    slot["center"] = (
                        0.7 * slot["center"][0] + 0.3 * c[0],
                        0.7 * slot["center"][1] + 0.3 * c[1],
                    )
                    slot["box"] = list(p["box"])
                    if p["signature"] is not None:
                        slot["signature"] = p["signature"]
                    slot["last_seen"] = self.frame_number
                    p["student_id"] = nearest
                else:
                    sid = self.next_student_id
                    self.next_student_id += 1
                    self.slots[sid] = {
                        "center": c,
                        "box": list(p["box"]),
                        "signature": p["signature"],
                        "last_seen": self.frame_number,
                    }
                    p["student_id"] = sid
            return persons


        candidates = []
        for pi, p in enumerate(persons):
            for sid, slot in self.slots.items():
                if self.frame_number - slot["last_seen"] > SLOT_MEMORY_FRAMES:
                    continue
                cost = self._assignment_cost(p, slot)
                if cost is not None:
                    candidates.append((cost, pi, sid))

        candidates.sort(key=lambda x: x[0])
        used_p, used_s = set(), set()

        for cost, pi, sid in candidates:
            if pi in used_p or sid in used_s:
                continue
            persons[pi]["student_id"] = sid
            used_p.add(pi)
            used_s.add(sid)



        for pi, p in enumerate(persons):
            if "student_id" not in p:
                sid = self.next_student_id
                self.next_student_id += 1
                self.slots[sid] = {
                    "center": self.center(p["box"]),
                    "box": list(p["box"]),
                    "signature": p["signature"],
                    "last_seen": self.frame_number,
                }
                p["student_id"] = sid


        for p in persons:
            sid = p["student_id"]
            slot = self.slots[sid]
            c = self.center(p["box"])
            slot["center"] = (
                0.80 * slot["center"][0] + 0.20 * c[0],
                0.80 * slot["center"][1] + 0.20 * c[1],
            )
            slot["box"] = list(p["box"])
            slot["last_seen"] = self.frame_number


            sig = p.get("signature")
            if sig is not None:
                if slot.get("signature") is None:
                    slot["signature"] = sig
                else:
                    slot["signature"] = (
                        0.90 * slot["signature"] + 0.10 * sig
                    )

        return persons

    @staticmethod
    def point_inside(point, box, padding=0):
        x, y = point
        x1, y1, x2, y2 = box
        return (x1-padding <= x <= x2+padding and
                y1-padding <= y <= y2+padding)

    def process_video(self, video_path, output_path=None,
                      progress_callback=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

        alerts_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self.frame_number += 1
            video_time = self.frame_number / fps

            result = self.model.track(
                frame,
                persist=True,
                tracker=str(BASE_DIR / "botsort_exam.yaml"),
                classes=[PERSON_CLASS, PHONE_CLASS],
                conf=0.35,
                verbose=False,
            )[0]

            persons, phones = [], []

            if result.boxes is not None:
                boxes = result.boxes
                raw_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )
                classes = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()

                for box, cls_id, conf, raw_id in zip(
                    coords, classes, confs, raw_ids
                ):
                    if cls_id == PERSON_CLASS:
                        persons.append({
                            "box": box,
                            "conf": conf,
                            "raw_id": raw_id,
                        })
                    elif cls_id == PHONE_CLASS:
                        phones.append({
                            "box": box,
                            "conf": conf,
                        })

            persons = self.update_slots(persons, frame)


            suspicious = {}
            for phone in phones:
                pc = self.center(phone["box"])
                best = None
                best_dist = 1e9

                for p in persons:
                    sid = p["student_id"]
                    box = p["box"]
                    scale = self._slot_scale(box)
                    inside = self.point_inside(
                        pc, box, padding=max(25, 0.15*scale)
                    )
                    d = self.distance(pc, self.center(box)) / scale

                    if inside or d < 0.75:
                        if d < best_dist:
                            best_dist = d
                            best = p

                if best is not None:
                    sid = best["student_id"]
                    suspicious[sid] = max(
                        suspicious.get(sid, 0.0),
                        phone["conf"]
                    )


            for phone in phones:
                x1, y1, x2, y2 = map(int, phone["box"])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(
                    frame, f"PHONE {phone['conf']:.2f}",
                    (x1, max(20,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2
                )

            active_count = 0

            for p in persons:
                sid = p["student_id"]
                raw = p["raw_id"]
                x1, y1, x2, y2 = map(int, p["box"])

                state = self.events.setdefault(
                    sid,
                    {
                        "active": False,
                        "streak": 0,
                        "last_suspicious_frame": -10**9,
                        "last_event_time": -10**9,
                    }
                )

                if sid in suspicious:
                    conf = suspicious[sid]
                    state["last_event_time"] = video_time

                    if self.frame_number == state["last_suspicious_frame"] + 1:
                        state["streak"] += 1
                    else:
                        state["streak"] = 1

                    state["last_suspicious_frame"] = self.frame_number

                    if (not state["active"] and
                            state["streak"] >= START_CONFIRM_FRAMES):
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        screenshot = (
                            SCREENSHOT_DIR /
                            f"cheating_{ts}_student_{sid}.jpg"
                        )

                        cv2.rectangle(
                            frame, (x1,y1), (x2,y2),
                            (0,0,255), 3
                        )
                        cv2.putText(
                            frame,
                            f"CHEATING | ID {sid} | Phone {conf:.2f}",
                            (x1, max(25,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0,0,255), 2
                        )
                        cv2.imwrite(str(screenshot), frame)

                        insert_alert(
                            behavior="Phone detected",
                            confidence=conf,
                            screenshot_path=screenshot,
                            student_track_id=sid
                        )
                        send_cheating_notification(
                            "Phone detected", conf
                        )

                        state["active"] = True
                        alerts_count += 1

                    if state["active"]:
                        active_count += 1

                else:
                    gap = self.frame_number - state["last_suspicious_frame"]
                    if state["active"]:
                        if gap <= END_GRACE_FRAMES:
                            active_count += 1
                        elif (video_time - state["last_event_time"]
                              <= INCIDENT_COOLDOWN_SECONDS):
                            active_count += 1
                        else:
                            state["active"] = False
                            state["streak"] = 0

                if state["active"]:
                    color = (0,0,255)
                    label = f"CHEATING | Student ID {sid}"
                else:
                    color = (0,255,0)
                    label = f"Student ID: {sid}"

                cv2.rectangle(
                    frame, (x1,y1), (x2,y2), color, 2
                )
                cv2.putText(
                    frame, label,
                    (x1, max(25,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2
                )


                cv2.putText(
                    frame, f"raw:{raw}",
                    (x1, min(height-8, y2+18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (80,80,80), 1
                )

            status = (
                f"ALERT ACTIVE | {active_count} suspicious student(s)"
                if active_count else "STATUS: NORMAL"
            )
            status_color = (0,0,255) if active_count else (0,180,0)

            cv2.rectangle(
                frame, (10,10), (540,52), (255,255,255), -1
            )
            cv2.putText(
                frame, status, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                status_color, 2
            )

            if writer:
                writer.write(frame)

            if progress_callback and total:
                progress_callback(self.frame_number / total)

        cap.release()
        if writer:
            writer.release()

        return alerts_count
