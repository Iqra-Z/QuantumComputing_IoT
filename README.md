# Quantum-Inspired Distracted Driving Detection on Raspberry Pi + Coral EdgeTPU

## Overview
This project demonstrates a real-time computer vision system running on:

- Raspberry Pi 5
- Raspberry Pi Camera (IMX708)
- Coral USB EdgeTPU

The system performs distracted driving detection and compares:

- Classical exhaustive decision search O(N)
- Quantum-inspired Grover-style search O(√N)

Both approaches run live and print performance metrics for comparison.  
This is a quantum-inspired algorithm running on classical hardware.

---

## Hardware Required

- Raspberry Pi 5
- Raspberry Pi Camera (connected via ribbon cable)
- Coral USB EdgeTPU
- Raspberry Pi OS (Desktop recommended)

---

## Software Requirements

- Raspberry Pi OS (Trixie / Bookworm)
- Python 3.9 (required for Coral)
- EdgeTPU runtime

---

## How to Run This

### Step 1 – Install System Dependencies

Run the following commands in terminal:

    sudo apt update
    sudo apt install -y python3-opencv libatlas-base-dev

Install EdgeTPU runtime:

    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    sudo apt install -y libedgetpu1-std

---

### Step 2 – Create Coral Python Environment

Coral requires Python 3.9.

Using pyenv:

    pyenv install 3.9.19
    pyenv virtualenv 3.9.19 coral39
    pyenv activate coral39

Install required Python packages:

    pip install "numpy<2" pillow pycoral tflite-runtime

Do NOT install pip OpenCV inside this environment (causes NumPy conflicts).

---

### Step 3 – Download Model Files

Place the following files inside:

    /home/iqra/coral_models/

Required files:

- model.tflite
- labels.txt

Adjust paths in run commands if needed.

---

### Step 4 – Run the System

You need TWO terminals.

#### Terminal 1 – Start Camera

    cd ~/capstone_demo
    /usr/bin/python3 cam_producer.py --preview --shm capstone_cam

This will:

- Start the camera
- Open preview window
- Send frames to shared memory

Leave this terminal running.

---

#### Terminal 2 – Activate Coral Environment and Run Detection

    cd ~/capstone_demo
    pyenv activate coral39

    python main.py --mode distracted \
    --model /home/iqra/coral_models/model.tflite \
    --labels /home/iqra/coral_models/labels.txt \
    --thresh 0.15 \
    --shm capstone_cam \
    --stable_window 5 \
    --debug \
    --run_tag test1

---

## System Behaviour

- Person only → FOCUSED
- Person + phone → DISTRACTED
- No person detected → UNKNOWN

A rolling 5-frame stability window is applied:

- raw decision → stable decision
- Prevents flickering between states
- Produces smoother real-time output

---

## Output Metrics

Every second the system prints:

- FPS
- Capture latency
- Preprocessing time
- TPU inference time
- Classical search time (64 evaluations)
- Grover-style search time (~8 evaluations)
- Speedup factor

Example output:

    CLASSICAL (O(N))   evals=64
    GROVER (~√N)       evals=8
    Speedup ≈ 7×

---

## Validation Testing

Structured test cases recorded:

- focused1
- focused2
- phone1
- phone2
- edge1
- edge2

Each run:

- ~60 seconds
- Real-time inference (~26–27 FPS)
- Stable decision output
- JSON logging enabled

Logs are stored in:

    ~/capstone_demo/logs/

Each JSONL file contains:

- FPS
- Detection count
- Raw decision
- Stable decision
- TPU inference metrics
- Classical vs Grover evaluation counts
- Decision latency
- Speedup ratio

---

## How It Works

1. Camera captures frame  
2. EdgeTPU runs object detection  
3. Scene encoded into a 6-bit logical state (0–63)  
4. Classical search checks all 64 states  
5. Grover-style search checks ~8 states  
6. Results compared live  
7. Stability window applied  

---

## Important Note

- This project does NOT use quantum hardware.
- It demonstrates a quantum-inspired decision search pattern implemented on embedded hardware.
- The quantum-inspired layer reduces oracle evaluations from 64 → ~8.
- Average observed decision-layer speedup ≈ 7×.

---

## To Stop

- Press Ctrl + C in the inference terminal
- Stop the camera terminal when finished
