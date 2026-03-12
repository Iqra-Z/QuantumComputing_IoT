#!/usr/bin/env python3

import time
import struct
import argparse
import random
import math
import os
import json
import datetime
from collections import deque
from multiprocessing import shared_memory

import requests
import cv2
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

# =========================
# TFLite detection helper (replacement for pycoral detect)
# =========================

class Detection:
    def __init__(self, bbox, score, class_id):
        self.bbox = bbox
        self.score = score
        self.id = class_id


class BBox:
    def __init__(self, ymin, xmin, ymax, xmax):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax


def get_objects(interpreter, threshold=0.2):

    output_details = interpreter.get_output_details()

    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    class_ids = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    count = int(interpreter.get_tensor(output_details[3]["index"])[0])

    objs = []

    for i in range(count):

        score = float(scores[i])
        if score < threshold:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        bbox = BBox(
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax
        )

        objs.append(
            Detection(
                bbox=bbox,
                score=score,
                class_id=int(class_ids[i])
            )
        )

    return objs



# =========================
# Grafana / dashboard
# =========================

# Put your real values here OR export them in the shell before running:
# export GRAFANA_URL="https://...."
# export GRAFANA_USER="...."
# export GRAFANA_TOKEN="...."

GRAFANA_URL = os.getenv("GRAFANA_URL", "")
GRAFANA_USER = os.getenv("GRAFANA_USER", "")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN", "")


def push_metrics_to_grafana(record: dict):
    if not GRAFANA_URL or not GRAFANA_USER or not GRAFANA_TOKEN:
        return None

    mode = str(record.get("mode", "unknown"))

    int_fields = {
        "fps": record.get("fps"),
        "dets": record.get("dets"),
        "has_person": 1 if record.get("has_person") else 0,
        "has_phone": 1 if record.get("has_phone") else 0,
        "has_driver_face": 1 if record.get("has_driver_face") else 0,
        "classical_evals": record.get("classical", {}).get("evals"),
        "classical_best_state": record.get("classical", {}).get("best_state"),
        "classical_score": record.get("classical", {}).get("score"),
        "grover_evals": record.get("grover", {}).get("evals"),
        "grover_best_state": record.get("grover", {}).get("best_state"),
        "grover_score": record.get("grover", {}).get("score"),
    }

    float_fields = {
        "frame_age_ms": record.get("frame_age_ms"),
        "pre_ms": record.get("pre_ms"),
        "tpu_infer_avg_ms": record.get("tpu_infer_avg_ms"),
        "phone_conf": record.get("phone_conf"),
        "driver_face_conf": record.get("driver_face_conf"),
        "speedup": record.get("speedup"),
        "classical_decision_ms": record.get("classical", {}).get("decision_ms"),
        "grover_decision_ms": record.get("grover", {}).get("decision_ms"),
    }

    fields = []

    for name, value in int_fields.items():
        if value is None:
            continue
        try:
            fields.append(f"{name}={int(value)}i")
        except (ValueError, TypeError):
            continue

    for name, value in float_fields.items():
        if value is None:
            continue
        try:
            fields.append(f"{name}={float(value)}")
        except (ValueError, TypeError):
            continue

    if not fields:
        return None

    ts_ns = int(record.get("ts_unix", time.time()) * 1e9)
    line = f"capstone_inference,mode={mode} {','.join(fields)} {ts_ns}"

    try:
        response = requests.post(
            GRAFANA_URL,
            headers={"Content-Type": "text/plain"},
            data=line,
            auth=(GRAFANA_USER, GRAFANA_TOKEN),
            timeout=5,
        )
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(f"[GRAFANA] Push failed: {e}")
        return None


# =========================
# Quantum-inspired decision layer
# =========================

N = 64
SQRT_N = 8


def popcount(x: int) -> int:
    return bin(x).count("1")


def hamming(a: int, b: int) -> int:
    return popcount(a ^ b)


def oracle_score(candidate: int, observed: int) -> int:
    return 6 - hamming(candidate, observed)


def classical_search(observed: int):
    best, best_score = 0, -999
    evals = 0
    t0 = time.perf_counter()
    for s in range(N):
        sc = oracle_score(s, observed)
        evals += 1
        if sc > best_score:
            best_score, best = sc, s
    t1 = time.perf_counter()
    return best, best_score, evals, (t1 - t0) * 1000.0


def grover_style_search(observed: int, seed: int):
    rng = random.Random(seed)
    candidates = set([observed])

    for _ in range(5):
        flips = rng.choice([1, 1, 2])
        bits = rng.sample(range(6), flips)
        s = observed
        for b in bits:
            s ^= (1 << b)
        candidates.add(s)

    while len(candidates) < SQRT_N:
        candidates.add(rng.randrange(N))

    best, best_score = 0, -999
    evals = 0
    t0 = time.perf_counter()
    for s in candidates:
        sc = oracle_score(s, observed)
        evals += 1
        if sc > best_score:
            best_score, best = sc, s
    t1 = time.perf_counter()
    return best, best_score, evals, (t1 - t0) * 1000.0


# =========================
# Labels
# =========================

def load_labels(path: str):
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        labels[int(parts[0])] = parts[1]
                else:
                    labels[len(labels)] = line
    except Exception:
        pass
    return labels


def name_of(labels, i: int) -> str:
    return labels.get(i, f"id_{i}")


# =========================
# Feature encoding
# =========================

def encode_state_distracted(has_phone, has_person, num_dets, phone_conf):
    # bits: phone, person, >=2 dets, strong phone conf, busy scene, reserved
    s = 0
    s |= (1 if has_phone else 0) << 0
    s |= (1 if has_person else 0) << 1
    s |= (1 if num_dets >= 2 else 0) << 2
    s |= (1 if phone_conf >= 0.60 else 0) << 3
    s |= (1 if num_dets >= 4 else 0) << 4
    s |= 0 << 5
    return s


def decision_label_distracted(has_person: bool, has_phone: bool, top_object: str):
    if not has_person:
        return "UNKNOWN", "NO_PERSON"
    if has_phone:
        return "DISTRACTED", "PHONE"
    if top_object:
        return "DISTRACTED", top_object.upper()
    return "FOCUSED", "OK"


def encode_state_count(num_vehicles, rate_per_min):
    s = 0
    s |= (1 if num_vehicles > 0 else 0) << 0
    s |= (1 if num_vehicles >= 2 else 0) << 1
    s |= (1 if num_vehicles >= 4 else 0) << 2
    s |= (1 if rate_per_min >= 5 else 0) << 3
    s |= (1 if rate_per_min >= 10 else 0) << 4
    s |= 0 << 5
    return s


def decision_label_count(state: int):
    busy = (state >> 2) & 1
    fast = (state >> 4) & 1
    anyv = state & 1
    if fast:
        return "TRAFFIC_HIGH", "RATE>=10/min"
    if busy:
        return "TRAFFIC_MED", "MANY_VEHICLES"
    if anyv:
        return "TRAFFIC_LOW", "SOME_VEHICLES"
    return "NO_TRAFFIC", "NONE"


# =========================
# Simple tracker for count mode
# =========================

class Track:
    __slots__ = ("id", "cx", "cy", "last_seen", "counted")

    def __init__(self, tid, cx, cy, t):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.last_seen = t
        self.counted = False


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


# =========================
# Distracted-driving helpers
# =========================

DISTRACT_OBJECT_CLASSES = {
    "cell phone",
    "cup",
    "bottle",
    "fork",
    "spoon",
    "banana",
    "sandwich",
    "pizza",
    "apple",
    "orange",
    "hot dog",
    "donut",
    "cake",
}

PERSON_NAMES = {"person"}
PHONE_NAMES = {"cell phone", "phone"}

VEHICLE_KEYWORDS = set()

HEADER_BYTES = 8 + 8 + 4 + 4 + 4 + 4


def clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))


def bbox_area(bb):
    return max(0.0, float(bb.xmax - bb.xmin)) * max(0.0, float(bb.ymax - bb.ymin))


def bbox_to_xyxy(bb, w, h):
    x1 = clamp_int(bb.xmin, 0, w - 1)
    y1 = clamp_int(bb.ymin, 0, h - 1)
    x2 = clamp_int(bb.xmax, 0, w - 1)
    y2 = clamp_int(bb.ymax, 0, h - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_box(x1, y1, x2, y2, pad_x, pad_y, w, h):
    return (
        clamp_int(x1 - pad_x, 0, w - 1),
        clamp_int(y1 - pad_y, 0, h - 1),
        clamp_int(x2 + pad_x, 0, w - 1),
        clamp_int(y2 + pad_y, 0, h - 1),
    )


def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def detect_faces_in_roi(face_cascade, frame_rgb, roi_box):
    if face_cascade is None:
        return []

    x1, y1, x2, y2 = roi_box
    if x2 <= x1 or y2 <= y1:
        return []

    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )

    out = []
    for (fx, fy, fw, fh) in faces:
        out.append((x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh))
    return out


def choose_driver_person(person_objs, frame_rgb, face_cascade, in_w, in_h):
    """
    Priority:
    1) person ROI containing the largest detected face
    2) fallback to largest person box
    """
    if not person_objs:
        return None, None, 0.0

    best_person = None
    best_face = None
    best_face_area = 0.0

    for obj in person_objs:
        pb = bbox_to_xyxy(obj.bbox, in_w, in_h)
        faces = detect_faces_in_roi(face_cascade, frame_rgb, pb)

        if faces:
            largest_face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            fa = float((largest_face[2] - largest_face[0]) * (largest_face[3] - largest_face[1]))
            if fa > best_face_area:
                best_face_area = fa
                best_person = obj
                best_face = largest_face

    if best_person is not None:
        return best_person, best_face, best_face_area

    # fallback: choose person closest to center of frame
    cx = in_w / 2
    cy = in_h / 2

    def center_distance(obj):
        bb = obj.bbox
        ox = (bb.xmin + bb.xmax) / 2
        oy = (bb.ymin + bb.ymax) / 2
        return (ox - cx) ** 2 + (oy - cy) ** 2

    best_person = min(person_objs, key=center_distance)
    return best_person, None, 0.0


def build_driver_focus_region(driver_obj, driver_face_box, in_w, in_h):
    """
    Main region used to decide whether an object belongs to the driver.
    If face exists, expand around face.
    Else expand around person box.
    """
    if driver_face_box is not None:
        fx1, fy1, fx2, fy2 = driver_face_box
        fw = fx2 - fx1
        fh = fy2 - fy1

        return expand_box(
            fx1,
            fy1,
            fx2,
            fy2,
            pad_x=int(1.4 * fw),
            pad_y=int(1.8 * fh),
            w=in_w,
            h=in_h,
        )

    dx1, dy1, dx2, dy2 = bbox_to_xyxy(driver_obj.bbox, in_w, in_h)
    dw = dx2 - dx1
    dh = dy2 - dy1

    return expand_box(
        dx1,
        dy1,
        dx2,
        dy2,
        pad_x=int(0.10 * dw),
        pad_y=int(0.10 * dh),
        w=in_w,
        h=in_h,
    )


def normalize_label(label: str) -> str:
    return label.strip().lower()


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distracted", "count"], default="distracted")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--thresh", type=float, default=0.20)
    ap.add_argument("--shm", default="capstone_cam")
    ap.add_argument("--line", type=float, default=0.60, help="counting: horizontal line position (0..1 of height)")
    ap.add_argument("--debug", action="store_true", help="print top classes once/sec")

    ap.add_argument("--stable_window", type=int, default=5, help="rolling window majority vote size")
    ap.add_argument("--face_cascade", default="haarcascade_frontalface_default.xml")
    ap.add_argument("--face_min_area", type=float, default=300.0, help="minimum face box area to trust")
    ap.add_argument("--driver_overlap_expand", type=float, default=0.35, help="expand object boxes slightly before overlap test")

    ap.add_argument("--log_dir", default="logs", help="directory to write JSONL logs")
    ap.add_argument("--run_tag", default="", help="optional tag for filename")
    ap.add_argument("--no_log", action="store_true", help="disable JSON logging")

    args = ap.parse_args()

    labels = load_labels(args.labels) if args.labels else {}

    stable_buf = deque(maxlen=max(1, args.stable_window))

    log_f = None
    log_path = None
    if not args.no_log:
        os.makedirs(args.log_dir, exist_ok=True)
        tag = args.run_tag.strip() or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"run_{tag}_{args.mode}.jsonl")
        log_f = open(log_path, "a", encoding="utf-8")
        print(f"[LOG] Writing JSONL metrics to: {log_path}")

    face_cascade = None
    if os.path.exists(args.face_cascade):
        try:
            face_cascade = cv2.CascadeClassifier(args.face_cascade)
            if face_cascade.empty():
                face_cascade = None
                print(f"[WARN] Failed to load face cascade: {args.face_cascade}")
        except Exception:
            face_cascade = None
            print(f"[WARN] Failed to initialize face cascade: {args.face_cascade}")
    else:
        print(f"[WARN] Face cascade not found: {args.face_cascade}")

    shm = shared_memory.SharedMemory(name=args.shm, create=False)
    buf = shm.buf

    interpreter = Interpreter(
        model_path=args.model,
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    in_h = input_shape[1]
    in_w = input_shape[2]

    last_frame_id = 0

    sec_start = time.time()
    frames = 0
    infer_sum = 0.0
    infer_n = 0
    last_pre_ms = 0.0
    last_frame_age_ms = 0.0

    next_id = 1
    tracks = []
    total_count = 0
    count_history = []

    try:
        while True:
            frame_id, ts_ns, w, h, c, _pad = struct.unpack_from("<QQIIII", buf, 0)
            if frame_id == 0 or frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = frame_id
            frames += 1

            frame_bytes = int(w) * int(h) * int(c)
        
            # skip invalid metadata
            if w == 0 or h == 0 or c == 0:
                continue

            raw = np.frombuffer(buf, dtype=np.uint8, offset=HEADER_BYTES, count=frame_bytes).copy()

            # skip partial frames
            if raw.size != frame_bytes:
                continue

            frame = raw.reshape((h, w, c))            
                        
            
            

            t_pre0 = time.perf_counter()
            img = Image.fromarray(frame).resize((in_w, in_h))

            resized = np.asarray(img)
            t_pre1 = time.perf_counter()
            last_pre_ms = (t_pre1 - t_pre0) * 1000.0

            try:
                last_frame_age_ms = (time.time_ns() - ts_ns) / 1e6
            except Exception:
                last_frame_age_ms = 0.0

            t_inf0 = time.perf_counter()
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(
                input_details[0]['index'],
                np.expand_dims(resized, axis=0).astype(input_details[0]['dtype'])
            )

            interpreter.invoke()
            t_inf1 = time.perf_counter()

            infer_ms = (t_inf1 - t_inf0) * 1000.0
            infer_sum += infer_ms
            infer_n += 1

            raw_objs = get_objects(interpreter, args.thresh)

            class_counts = {}
            top_classes = []

            has_person = False
            has_phone = False
            phone_conf = 0.0
            top_object = ""
            driver_face_conf = 0.0
            has_driver_face = False
            driver_box = None
            driver_face_box = None
            driver_focus_region = None

            vehicles = []

            observed = 0
            dec = "UNKNOWN"
            dec_raw = "UNKNOWN"
            reason = "INIT"
            extra = ""

            if args.mode == "distracted":
                person_objs = []
                candidate_objs = []

                for o in raw_objs:
                    nm = normalize_label(name_of(labels, o.id))
                    if not nm:
                        continue

                    class_counts[o.id] = class_counts.get(o.id, 0) + 1

                    if nm in PERSON_NAMES:
                        person_objs.append(o)
                    elif nm in DISTRACT_OBJECT_CLASSES or nm in PHONE_NAMES:
                        candidate_objs.append((o, nm))

                if class_counts:
                    top = sorted(class_counts.items(), key=lambda x: -x[1])[:6]
                    for k, v in top:
                        top_classes.append({
                            "id": int(k),
                            "name": name_of(labels, k),
                            "count": int(v),
                        })

                driver_obj, driver_face_box, driver_face_area = choose_driver_person(
                    person_objs,
                    resized,
                    face_cascade,
                    in_w,
                    in_h,
                )

                if driver_obj is not None:
                    has_person = True
                    driver_box = bbox_to_xyxy(driver_obj.bbox, in_w, in_h)

                if driver_face_box is not None and driver_face_area >= args.face_min_area:
                    has_driver_face = True
                    driver_face_conf = float(driver_face_area)

                if driver_obj is not None:
                    driver_focus_region = build_driver_focus_region(
                        driver_obj,
                        driver_face_box if has_driver_face else None,
                        in_w,
                        in_h,
                    )

                relevant_driver_objs = []
                relevant_labels = []

                if driver_obj is not None:
                    for obj, nm in candidate_objs:
                        ob = bbox_to_xyxy(obj.bbox, in_w, in_h)

                        ox1, oy1, ox2, oy2 = ob
                        ow = max(1, ox2 - ox1)
                        oh = max(1, oy2 - oy1)
                        expanded_obj_box = expand_box(
                            ox1,
                            oy1,
                            ox2,
                            oy2,
                            pad_x=int(0.15 * ow),
                            pad_y=int(0.15 * oh),
                            w=in_w,
                            h=in_h,
                        )

                        if boxes_intersect(expanded_obj_box, driver_focus_region):
                            relevant_driver_objs.append(obj)
                            relevant_labels.append(nm)

                            if nm in PHONE_NAMES:
                                has_phone = True
                                phone_conf = max(phone_conf, float(obj.score))

                    non_phone_objs = [(obj, nm) for obj, nm in zip(relevant_driver_objs, relevant_labels) if nm not in PHONE_NAMES]
                    if non_phone_objs:
                        best_obj, best_nm = max(non_phone_objs, key=lambda t: float(t[0].score))
                        top_object = best_nm

                num_dets_for_state = (1 if has_person else 0) + len(relevant_driver_objs)

                observed = encode_state_distracted(
                    has_phone=has_phone,
                    has_person=has_person,
                    num_dets=num_dets_for_state,
                    phone_conf=phone_conf,
                )

                dec_raw, reason = decision_label_distracted(
                    has_person=has_person,
                    has_phone=has_phone,
                    top_object=top_object,
                )

                stable_buf.append(dec_raw)
                need = (len(stable_buf) // 2) + 1
                distracted_votes = sum(1 for x in stable_buf if x == "DISTRACTED")
                unknown_votes = sum(1 for x in stable_buf if x == "UNKNOWN")

                if unknown_votes >= need:
                    dec = "UNKNOWN"
                elif distracted_votes >= need:
                    dec = "DISTRACTED"
                else:
                    dec = "FOCUSED"

                if has_phone:
                    extra = f"phone_conf={phone_conf:.2f}"
                elif top_object:
                    extra = f"object={top_object}"
                else:
                    extra = ""

                dets_for_print = num_dets_for_state

            else:
                for o in raw_objs:
                    nm = normalize_label(name_of(labels, o.id))
                    class_counts[o.id] = class_counts.get(o.id, 0) + 1

                    if any(k in nm for k in VEHICLE_KEYWORDS):
                        bb = o.bbox
                        cx = (bb.xmin + bb.xmax) / 2.0
                        cy = (bb.ymin + bb.ymax) / 2.0
                        vehicles.append((cx, cy))

                if class_counts:
                    top = sorted(class_counts.items(), key=lambda x: -x[1])[:6]
                    for k, v in top:
                        top_classes.append({
                            "id": int(k),
                            "name": name_of(labels, k),
                            "count": int(v),
                        })

                tnow = time.time()
                line_y = args.line * in_h

                used = set()
                for tr in tracks:
                    best_j = -1
                    best_d = 999999.0
                    for j, (cx, cy) in enumerate(vehicles):
                        if j in used:
                            continue
                        d = dist(tr.cx, tr.cy, cx, cy)
                        if d < best_d:
                            best_d, best_j = d, j
                    if best_j != -1 and best_d < 60:
                        tr.cx, tr.cy = vehicles[best_j]
                        tr.last_seen = tnow
                        used.add(best_j)

                for j, (cx, cy) in enumerate(vehicles):
                    if j not in used:
                        tracks.append(Track(next_id, cx, cy, tnow))
                        next_id += 1

                tracks = [tr for tr in tracks if (tnow - tr.last_seen) < 0.8]

                for tr in tracks:
                    if not tr.counted and tr.cy > line_y:
                        tr.counted = True
                        total_count += 1
                        count_history.append(tnow)

                count_history = [t for t in count_history if (tnow - t) <= 60.0]
                rate = len(count_history)

                observed = encode_state_count(len(vehicles), rate)
                dec, reason = decision_label_count(observed)
                dec_raw = ""
                extra = f"vehicles={len(vehicles)} total={total_count} rate={rate}/min"
                dets_for_print = len(vehicles)

            now = time.time()
            if now - sec_start >= 1.0:
                avg_inf = infer_sum / max(1, infer_n)

                best_c, score_c, evals_c, lat_c = classical_search(observed)
                best_g, score_g, evals_g, lat_g = grover_style_search(observed, seed=frame_id)
                speedup = (lat_c / lat_g) if lat_g > 0 else 0.0

                record = {
                    "ts_unix": float(now),
                    "frame_id": int(frame_id),
                    "mode": args.mode,
                    "fps": int(frames),
                    "dets": int(dets_for_print),
                    "decision": dec,
                    "decision_raw": dec_raw,
                    "reason": reason,
                    "extra": extra,
                    "frame_age_ms": float(last_frame_age_ms),
                    "pre_ms": float(last_pre_ms),
                    "tpu_infer_avg_ms": float(avg_inf),
                    "has_person": bool(has_person),
                    "has_phone": bool(has_phone),
                    "has_driver_face": bool(has_driver_face),
                    "driver_face_conf": float(driver_face_conf),
                    "phone_conf": float(phone_conf),
                    "top_object": top_object,
                    "top_classes": top_classes,
                    "classical": {
                        "evals": int(evals_c),
                        "decision_ms": float(lat_c),
                        "best_state": int(best_c),
                        "score": int(score_c),
                    },
                    "grover": {
                        "evals": int(evals_g),
                        "decision_ms": float(lat_g),
                        "best_state": int(best_g),
                        "score": int(score_g),
                    },
                    "speedup": float(speedup),
                }

                if log_f is not None:
                    log_f.write(json.dumps(record) + "\n")
                    log_f.flush()

                push_metrics_to_grafana(record)

                print("\n" + "=" * 78)
                if args.mode == "distracted":
                    print(
                        f"MODE=DISTRACTED | FPS={frames:2d} | dets={dets_for_print:2d} | "
                        f"raw={dec_raw} -> stable={dec} ({reason}) {extra}"
                    )
                    print(
                        f"has_person={int(has_person)} | has_phone={int(has_phone)}"
                    )
                else:
                    print(
                        f"MODE=COUNT      | FPS={frames:2d} | dets={dets_for_print:2d} | "
                        f"{dec} ({reason}) {extra}"
                    )

                print(
                    f"Capture→Now: {last_frame_age_ms:7.1f} ms | "
                    f"Pre: {last_pre_ms:6.2f} ms | "
                    f"TPU infer avg: {avg_inf:6.2f} ms"
                )
                print("-" * 78)
                print(
                    f"CLASSICAL (O(N))   evals={evals_c:2d}  decision_ms={lat_c:7.3f}  "
                    f"best_state={best_c:02d}  score={score_c}"
                )
                print(
                    f"GROVER (~√N)       evals={evals_g:2d}  decision_ms={lat_g:7.3f}  "
                    f"best_state={best_g:02d}  score={score_g}"
                )
                print(
                    f"Speedup (decision layer): {speedup:5.2f}x   |   "
                    f"Oracle evals: {evals_c} vs ~{evals_g}"
                )

                if args.debug and top_classes:
                    top_s = ", ".join(
                        [f"{d['id']}:{d['name']}({d['count']})" for d in top_classes]
                    )
                    print(f"Top classes: {top_s}")

                print("=" * 78)

                sec_start = now
                frames = 0
                infer_sum = 0.0
                infer_n = 0

    finally:
        try:
            if log_f is not None:
                log_f.close()
        except Exception:
            pass
        shm.close()


if __name__ == "__main__":
    main()
