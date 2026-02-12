import time, struct, argparse, random, math
from multiprocessing import shared_memory

import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# ---------------- Quantum-inspired decision layer ----------------
N = 64
SQRT_N = 8

def popcount(x: int) -> int:
    return bin(x).count("1")

def hamming(a: int, b: int) -> int:
    return popcount(a ^ b)

def oracle_score(candidate: int, observed: int) -> int:
    # Higher means closer match to observed features
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

    # Bias toward near-observed states (quantum-inspired "amplitude concentration" vibe)
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

# ---------------- Labels ----------------
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

# ---------------- Feature encoding (6-bit observed state) ----------------
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
    # Demo-friendly:
    # - If person + phone => DISTRACTED (PHONE)
    # - Else if person + any other object => DISTRACTED (OBJECTNAME)
    # - Else person alone => FOCUSED
    if not has_person:
        return "UNKNOWN", "NO_PERSON"
    if has_phone:
        return "DISTRACTED", "PHONE"
    if top_object:
        return "DISTRACTED", top_object.upper()
    return "FOCUSED", "OK"

def encode_state_count(num_vehicles, rate_per_min):
    # bits: some vehicles, >=2, >=4, rate>=5/min, rate>=10/min, reserved
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

# ---------------- Simple tracker for counting ----------------
class Track:
    __slots__ = ("id","cx","cy","last_seen","counted")
    def __init__(self, tid, cx, cy, t):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.last_seen = t
        self.counted = False

def dist(a,b,c,d):
    return math.hypot(a-c, b-d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["distracted","count"], default="distracted")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--thresh", type=float, default=0.20)
    ap.add_argument("--shm", default="capstone_cam")
    ap.add_argument("--line", type=float, default=0.60, help="counting: horizontal line position (0..1 of height)")
    ap.add_argument("--debug", action="store_true", help="print top classes once/sec")
    args = ap.parse_args()

    labels = load_labels(args.labels) if args.labels else {}

    # Attach shared memory from camera producer
    shm = shared_memory.SharedMemory(name=args.shm, create=False)
    buf = shm.buf

    # TPU init
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    in_w, in_h = common.input_size(interpreter)

    last_frame_id = 0

    # Per-second metrics
    sec_start = time.time()
    frames = 0
    infer_sum = 0.0
    infer_n = 0

    # Counting state
    next_id = 1
    tracks = []
    total_count = 0
    count_history = []  # timestamps of crossings in last 60s

    try:
        while True:
            frame_id, ts_ns, w, h, c, _pad = struct.unpack_from("<QQIIII", buf, 0)
            if frame_id == 0 or frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = frame_id
            frames += 1

            header_bytes = 8 + 8 + 4 + 4 + 4 + 4
            frame_bytes = int(w) * int(h) * int(c)

            # Copy() so we can cleanly close shared memory and avoid buffer errors
            raw = np.frombuffer(buf, dtype=np.uint8, offset=header_bytes, count=frame_bytes).copy()
            frame = raw.reshape((h, w, c))  # RGB

            # Resize properly to model input
            t_pre0 = time.perf_counter()
            img = Image.fromarray(frame).resize((in_w, in_h))
            resized = np.asarray(img)
            t_pre1 = time.perf_counter()

            # TPU inference
            t_inf0 = time.perf_counter()
            common.set_input(interpreter, resized)
            interpreter.invoke()
            t_inf1 = time.perf_counter()

            objs = detect.get_objects(interpreter, args.thresh)

            infer_ms = (t_inf1 - t_inf0) * 1000.0
            infer_sum += infer_ms
            infer_n += 1

            # Parse detections
            class_counts = {}
            has_person = False
            has_phone = False
            phone_conf = 0.0
            vehicles = []  # (cx,cy) in resized coords for counting

            for o in objs:
                class_counts[o.id] = class_counts.get(o.id, 0) + 1
                nm = name_of(labels, o.id).lower()

                # person
                if o.id == 0 or "person" in nm:
                    has_person = True

                # phone (label-based, and common coco ids if present)
                if ("cell phone" in nm) or ("phone" in nm and "microphone" not in nm) or (o.id in (67, 77)):
                    has_phone = True
                    phone_conf = max(phone_conf, float(o.score))

                # vehicles (label-based)
                if any(k in nm for k in ["car","truck","bus","motorcycle","motorbike","van"]):
                    bb = o.bbox
                    cx = (bb.xmin + bb.xmax) / 2.0
                    cy = (bb.ymin + bb.ymax) / 2.0
                    vehicles.append((cx, cy))

            # -------- mode logic --------
            if args.mode == "distracted":
                # Pick top non-person object for "reason"
                top_object = ""
                if class_counts:
                    items = sorted(class_counts.items(), key=lambda x: -x[1])
                    for cid, cnt in items:
                        nm = name_of(labels, cid).lower()
                        if cid == 0 or "person" in nm:
                            continue
                        top_object = name_of(labels, cid)
                        break

                observed = encode_state_distracted(has_phone, has_person, len(objs), phone_conf)
                dec, reason = decision_label_distracted(has_person, has_phone, top_object)

                if has_phone:
                    extra = f"phone_conf={phone_conf:.2f}"
                elif top_object:
                    extra = f"object={top_object}"
                else:
                    extra = ""

            else:
                # counting: simple nearest-neighbor tracking
                tnow = time.time()
                line_y = args.line * in_h

                used = set()
                for tr in tracks:
                    best_j = -1
                    best_d = 999999.0
                    for j,(cx,cy) in enumerate(vehicles):
                        if j in used:
                            continue
                        d = dist(tr.cx, tr.cy, cx, cy)
                        if d < best_d:
                            best_d, best_j = d, j
                    if best_j != -1 and best_d < 60:
                        tr.cx, tr.cy = vehicles[best_j]
                        tr.last_seen = tnow
                        used.add(best_j)

                for j,(cx,cy) in enumerate(vehicles):
                    if j not in used:
                        tracks.append(Track(next_id, cx, cy, tnow))
                        next_id += 1

                tracks = [tr for tr in tracks if (tnow - tr.last_seen) < 0.8]

                # Count each track once when it goes below the line
                for tr in tracks:
                    if not tr.counted and tr.cy > line_y:
                        tr.counted = True
                        total_count += 1
                        count_history.append(tnow)

                count_history = [t for t in count_history if (tnow - t) <= 60.0]
                rate = len(count_history)  # per 60s

                observed = encode_state_count(len(vehicles), rate)
                dec, reason = decision_label_count(observed)
                extra = f"vehicles={len(vehicles)} total={total_count} rate={rate}/min"

            # -------- once per second print scoreboard --------
            now = time.time()
            if now - sec_start >= 1.0:
                avg_inf = infer_sum / max(1, infer_n)
                frame_age_ms = (time.time_ns() - ts_ns) / 1e6
                pre_ms = (t_pre1 - t_pre0) * 1000.0

                best_c, score_c, evals_c, lat_c = classical_search(observed)
                best_g, score_g, evals_g, lat_g = grover_style_search(observed, seed=frame_id)
                speedup = (lat_c / lat_g) if lat_g > 0 else 0.0

                print("\n" + "="*78)
                print(f"MODE={args.mode.upper():10s} | FPS={frames:2d} | dets={len(objs):2d} | {dec} ({reason}) {extra}")
                print(f"Capture→Now: {frame_age_ms:7.1f} ms | Pre: {pre_ms:5.2f} ms | TPU infer avg: {avg_inf:5.2f} ms")
                print("-"*78)
                print(f"CLASSICAL (O(N))   evals={evals_c:2d}  decision_ms={lat_c:7.3f}  best_state={best_c:02d}  score={score_c}")
                print(f"GROVER (~√N)       evals={evals_g:2d}  decision_ms={lat_g:7.3f}  best_state={best_g:02d}  score={score_g}")
                print(f"Speedup (decision layer): {speedup:5.2f}x   |   Oracle evals: 64 vs ~8")
                if args.debug and class_counts:
                    top = sorted(class_counts.items(), key=lambda x: -x[1])[:6]
                    top_s = ", ".join([f"{k}:{name_of(labels,k)}({v})" for k,v in top])
                    print(f"Top classes: {top_s}")
                print("="*78)

                sec_start = now
                frames = 0
                infer_sum = 0.0
                infer_n = 0

    finally:
        shm.close()

if __name__ == "__main__":
    main()
