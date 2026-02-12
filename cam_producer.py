#!/usr/bin/env python3
import time, struct, argparse
from multiprocessing import shared_memory
from picamera2 import Picamera2

SHM_NAME_DEFAULT = "capstone_cam"
W, H, C = 640, 480, 3
FRAME_BYTES = W * H * C
HEADER_BYTES = 8 + 8 + 4 + 4 + 4 + 4
SHM_BYTES = HEADER_BYTES + FRAME_BYTES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm", default=SHM_NAME_DEFAULT)
    ap.add_argument("--preview", action="store_true", help="Show preview window (press q to quit)")
    ap.add_argument("--flip", action="store_true", help="Mirror horizontally")
    args = ap.parse_args()

    try:
        shm = shared_memory.SharedMemory(name=args.shm, create=True, size=SHM_BYTES)
        print("[CAM] Created shared memory:", args.shm)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=args.shm, create=False, size=SHM_BYTES)
        print("[CAM] Attached to existing shared memory:", args.shm)

    buf = shm.buf

    picam2 = Picamera2()
    # We request RGB888, but on your setup R/B end up swapped.
    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (W, H)},
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(config)
    picam2.start()

    show = args.preview
    if show:
        import cv2
        win = "Capstone Camera Preview (press q)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_id = 0
    last_print = time.time()
    fps = 0

    try:
        while True:
            t0 = time.perf_counter_ns()

            frame = picam2.capture_array()  # claims RGB, but R/B are swapped on your system
            if args.flip:
                frame = frame[:, ::-1, :]

            # FIX: swap R <-> B once
            frame_rgb = frame[..., ::-1]  # now truly RGB

            t1 = time.perf_counter_ns()

            # Write TRUE RGB to shared memory for TPU
            buf[HEADER_BYTES:HEADER_BYTES + FRAME_BYTES] = frame_rgb.ravel().tobytes()

            frame_id += 1
            ts_ns = time.time_ns()
            struct.pack_into("<QQIIII", buf, 0, frame_id, ts_ns, W, H, C, 0)

            fps += 1
            now = time.time()
            if now - last_print >= 1.0:
                cap_ms = (t1 - t0) / 1e6
                print(f"[CAM] FPS={fps} capture_ms={cap_ms:.2f} frame_id={frame_id}")
                fps = 0
                last_print = now

            if show:
                import cv2
                # OpenCV needs BGR, so convert RGB->BGR for display
                frame_bgr = frame_rgb[..., ::-1]
                cv2.imshow(win, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        picam2.stop()
        shm.close()
        if show:
            import cv2
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
