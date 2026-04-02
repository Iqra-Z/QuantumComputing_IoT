"""
Configuration for the Raspberry Pi face-detection device agent.

All values can be overridden with environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# Identity and network
DEVICE_NAME = os.getenv("DEVICE_NAME", "QFocus")
PORT        = int(os.getenv("PORT", "8002"))

# Stream settings
TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))
SHM_NAME   = os.getenv("SHM_NAME", "capstone_cam")

# Inference settings
MODEL_PATH   = os.getenv("MODEL_PATH", "detect.tflite")
LABELS_PATH  = os.getenv("LABELS_PATH", "coco_labels.txt")
DETECT_MODE       = os.getenv("DETECT_MODE", "distracted")
DETECT_THRESHOLD  = float(os.getenv("DETECT_THRESHOLD", "0.12"))
FACE_CASCADE = os.getenv(
    "FACE_CASCADE",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
)
