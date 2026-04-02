"""
Background pipeline runner for the face-detection device agent.

Manages the lifecycle of:
  1. cam_producer.py  — captures Pi camera frames into shared memory
  2. main.py          — reads shared memory, runs TFLite inference, pushes to Grafana

The video feed is served by reading the same shared memory, avoiding
any camera access conflicts between processes.
"""

import json
import logging
import struct
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory
from typing import Iterator, Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

# Shared memory layout — must match cam_producer.py
HEADER_BYTES = 8 + 8 + 4 + 4 + 4 + 4  # frame_id, ts_ns, w, h, c, pad


def _build_placeholder_jpeg(message: str) -> bytes:
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        config.DEVICE_NAME,
        (24, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 220, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message[:72],
        (24, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        time.strftime("%Y-%m-%d %H:%M:%S"),
        (24, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes() if ok else b""


class PipelineStreamRunner:
    """Owns the lifecycle of the face-detection pipeline and MJPEG stream."""

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._running       = False
        self._last_error: Optional[str] = None
        self._cam_proc: Optional[subprocess.Popen] = None
        self._main_proc: Optional[subprocess.Popen] = None
        self._latest_jpeg: Optional[bytes] = None
        self._stop_event    = threading.Event()

        # Cached detection data — updated by background thread, read by frame encoder
        self._det_lock      = threading.Lock()
        self._cached_det: Optional[dict] = None

        # Sticky-box state: keep last detected face/phone visible for a short time
        # to absorb per-frame Haar noise (prevents flickering boxes).
        # Accessed only from _det_reader_loop thread — no extra lock needed.
        self._last_face: Optional[list] = None
        self._last_face_ts: float = 0.0
        self._last_phones: list = []
        self._last_phones_ts: float = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        with self._lock:
            if self._running:
                cam_dead  = self._cam_proc  is None or self._cam_proc.poll()  is not None
                main_dead = self._main_proc is None or self._main_proc.poll() is not None
                if cam_dead or main_dead:
                    self._running     = False
                    self._last_error  = "A pipeline process exited unexpectedly"
                    self._latest_jpeg = None
            return self._running

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return False, "Already running"
            self._last_error  = None
            self._latest_jpeg = None

        python = sys.executable

        # 1. Start cam_producer
        try:
            cam_proc = subprocess.Popen(
                [python, str(config.ROOT_DIR / "cam_producer.py")],
                cwd=str(config.ROOT_DIR),
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return False, f"Failed to start cam_producer: {exc}"

        # Give cam_producer time to create shared memory before main.py attaches
        time.sleep(1.5)

        # 2. Start main.py (inference + Grafana)
        try:
            main_proc = subprocess.Popen(
                [python, str(config.ROOT_DIR / "main.py"),
                 "--model",        config.MODEL_PATH,
                 "--labels",       config.LABELS_PATH,
                 "--mode",         config.DETECT_MODE,
                 "--thresh",       str(config.DETECT_THRESHOLD),
                 "--face_cascade", config.FACE_CASCADE],
                cwd=str(config.ROOT_DIR),
            )
        except Exception as exc:
            cam_proc.terminate()
            with self._lock:
                self._last_error = str(exc)
            return False, f"Failed to start main.py: {exc}"

        # 3. Start background threads
        self._stop_event.clear()
        threading.Thread(target=self._reader_loop, daemon=True, name="shm-reader").start()
        threading.Thread(target=self._det_reader_loop, daemon=True, name="det-reader").start()

        with self._lock:
            self._cam_proc  = cam_proc
            self._main_proc = main_proc
            self._running   = True

        log.info("Pipeline started (cam_producer pid=%d, main pid=%d)",
                 cam_proc.pid, main_proc.pid)
        return True, "Stream started"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self._running:
                return False, "Not running"
            cam_proc  = self._cam_proc
            main_proc = self._main_proc

        self._stop_event.set()

        for proc in (main_proc, cam_proc):
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        with self._lock:
            self._cam_proc    = None
            self._main_proc   = None
            self._running     = False
            self._latest_jpeg = None

        with self._det_lock:
            self._cached_det = None

        self._last_face = None
        self._last_face_ts = 0.0
        self._last_phones = []
        self._last_phones_ts = 0.0

        log.info("Pipeline stopped")
        return True, "Stopped"

    def mjpeg_chunks(self) -> Iterator[bytes]:
        delay = 1.0 / max(config.TARGET_FPS, 1)
        while True:
            frame = self._get_latest_or_placeholder()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
            time.sleep(delay)

    # ── internals ─────────────────────────────────────────────────────────────

    def _get_latest_or_placeholder(self) -> bytes:
        with self._lock:
            running = self._running
            err     = self._last_error
            latest  = self._latest_jpeg

        if err:
            return _build_placeholder_jpeg(f"ERROR: {err}")
        if running and latest is not None:
            return latest
        if running:
            return _build_placeholder_jpeg("Starting camera...")
        return _build_placeholder_jpeg("Stream offline. Press Start in dashboard.")

    def _det_reader_loop(self) -> None:
        """Background thread: polls latest_detections.json and caches result in memory.

        Sticky-box logic: if the current detection has no face/phone but we saw one
        recently (within STICKY_SECS), inject the last-known box so the overlay doesn't
        flicker every time the Haar cascade misses a single frame.
        """
        STICKY_SECS = 1.5
        det_path = config.ROOT_DIR / "latest_detections.json"
        while not self._stop_event.is_set():
            try:
                if det_path.exists():
                    with open(det_path) as f:
                        det = json.load(f)

                    now = time.time()
                    if now - det.get("ts", 0) > 3.0:
                        # Stale — clear cache
                        with self._det_lock:
                            self._cached_det = None
                    else:
                        # Update sticky state from fresh detection
                        if det.get("face") is not None:
                            self._last_face = det["face"]
                            self._last_face_ts = now
                        if det.get("objects"):
                            self._last_phones = det["objects"]
                            self._last_phones_ts = now

                        # Inject sticky face if current frame has none
                        if det.get("face") is None and (now - self._last_face_ts) < STICKY_SECS:
                            det = dict(det)
                            det["face"] = self._last_face

                        # Inject sticky phones if current frame has none
                        if not det.get("objects") and (now - self._last_phones_ts) < STICKY_SECS:
                            det = dict(det)
                            det["objects"] = self._last_phones

                        with self._det_lock:
                            self._cached_det = det
            except Exception:
                pass
            time.sleep(0.1)  # poll at 10 Hz

    def _draw_detections(self, frame: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
        """Draw bounding boxes from cached detection data onto frame.

        All box coordinates in the JSON are normalized (0–1); multiply by
        frame_w / frame_h to get pixel positions in the output frame.
        """
        with self._det_lock:
            det = self._cached_det

        if det is None:
            return frame

        # Face box — cyan
        face = det.get("face")
        if face:
            x1 = int(face[0] * frame_w)
            y1 = int(face[1] * frame_h)
            x2 = int(face[2] * frame_w)
            y2 = int(face[3] * frame_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "face", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # Phone boxes only — orange
        for box in det.get("objects", []):
            label = box[4] if len(box) > 4 else ""
            if label not in ("cell phone", "phone"):
                continue
            x1 = int(box[0] * frame_w)
            y1 = int(box[1] * frame_h)
            x2 = int(box[2] * frame_w)
            y2 = int(box[3] * frame_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # Decision banner at the top
        decision = det.get("decision", "")
        reason   = det.get("reason", "")
        color = (0, 0, 255) if decision == "DISTRACTED" else \
                (0, 255, 0) if decision == "FOCUSED" else (128, 128, 128)
        text = f"{decision}  {reason}" if reason else decision
        cv2.rectangle(frame, (0, 0), (frame_w, 26), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return frame

    def _reader_loop(self) -> None:
        """Background thread: reads frames from shared memory and encodes as JPEG.

        Exception handling is deliberately split:
          - SharedMemory / struct errors  → close & reopen SHM (hardware issue)
          - Drawing / encoding errors     → skip this frame, keep SHM open
                                            (a buggy overlay must not drop video)
        """
        shm: Optional[shared_memory.SharedMemory] = None
        last_frame_id = 0

        while not self._stop_event.is_set():
            # ── attach to shared memory ───────────────────────────────────────
            if shm is None:
                try:
                    shm = shared_memory.SharedMemory(name=config.SHM_NAME, create=False)
                except Exception:
                    time.sleep(0.2)
                    continue

            # ── read frame header ─────────────────────────────────────────────
            try:
                buf = shm.buf
                frame_id = struct.unpack_from("<Q", buf, 0)[0]

                if frame_id == 0 or frame_id == last_frame_id:
                    time.sleep(0.01)
                    continue

                _, _ts, w, h, c, _ = struct.unpack_from("<QQIIII", buf, 0)
                if w == 0 or h == 0 or c == 0:
                    time.sleep(0.01)
                    continue

                raw = np.frombuffer(buf, dtype=np.uint8,
                                    offset=HEADER_BYTES,
                                    count=int(w * h * c)).copy()

                # Verify frame_id hasn't changed during the copy (torn-frame guard).
                # cam_producer writes pixels THEN header, so a changed id means a new
                # frame started overwriting while we were copying — discard it.
                if struct.unpack_from("<Q", buf, 0)[0] != frame_id:
                    continue

                last_frame_id = frame_id

            except Exception as exc:
                log.warning("SHM read error: %s", exc)
                try:
                    shm.close()
                except Exception:
                    pass
                shm = None
                time.sleep(0.2)
                continue

            # ── draw detections and encode JPEG ───────────────────────────────
            # Errors here must NOT close the SHM — that would freeze the video.
            try:
                frame_rgb = raw.reshape((int(h), int(w), int(c)))
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                bgr = self._draw_detections(bgr, int(w), int(h))
                ok, encoded = cv2.imencode(
                    ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()
            except Exception as exc:
                log.warning("Frame encode error (SHM kept open): %s", exc)

        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
