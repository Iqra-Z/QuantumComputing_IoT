#Quantum-Inspired Decision Algorithms on Raspberry Pi + Coral EdgeTPU

##Overview
This project demonstrates a real-time computer vision system running on:
-   Raspberry Pi 5
-   Raspberry Pi Camera (IMX708)
-   Coral USB EdgeTPU

The system compares:
-   Classical exhaustive decision search O(N)
-   Quantum-inspired Grover-style search O(√N)

Both approaches run live and print performance metrics for comparison. This is a quantum-inspired algorithm running on classical hardware.

---

##Hardware Required
-   Raspberry Pi 5
-   Raspberry Pi Camera (connected via ribbon cable)
-   Coral USB EdgeTPU
-   Raspberry Pi OS (Desktop recommended)

##Software Requirements
-   Raspberry Pi OS (Trixie / Bookworm)
-   Python 3.9 (required for Coral)
-   EdgeTPU runtime

---

##How to Run This

###Step 1 – Install System Dependencies
Run the following commands in terminal:

    sudo apt update
    sudo apt install -y python3-opencv libatlas-base-dev

Install EdgeTPU runtime:

    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    sudo apt install -y libedgetpu1-std

###Step 2 – Create Coral Python Environment

Coral requires Python 3.9.

Using pyenv:

    pyenv install 3.9.19
    pyenv virtualenv 3.9.19 coral39
    pyenv activate coral39

Install required Python packages:

    pip install numpy<2 pillow pycoral tflite-runtime

-   Do NOT install pip OpenCV inside this environment (causes NumPy
    conflicts).

###Step 3 – Download Model Files

Place the following files inside a directory such as:

-   /home/pi/coral_models/

Required files:

-   model.tflite
-   labels.txt

Adjust paths in run commands if needed.

###Step 4 – Run the System

You need TWO terminals.

Terminal 1 – Start Camera

    cd ~/capstone_demo
    /usr/bin/python3 cam_producer.py --preview

This will: - Start the camera - Open preview window - Send frames to
shared memory

Leave this terminal running.

Terminal 2 – Activate Coral Environment

    cd ~/capstone_demo
    pyenv activate coral39

Mode 1 – Distracted Driving

    python main.py --mode distracted --model /home/pi/coral_models/model.tflite --labels /home/pi/coral_models/labels.txt --thresh 0.15 --shm capstone_cam --debug

Behavior:

-   Person only → FOCUSED
-   Person + phone/object → DISTRACTED
-   Classical vs Grover comparison displayed live

Mode 2 – Car Counting

Stop distracted mode with Ctrl + C.

Then run:

    python main.py --mode count --model /home/pi/coral_models/model.tflite --labels /home/pi/coral_models/labels.txt --thresh 0.15 --shm capstone_cam --line 0.60 --debug

##Behavior:

-   Detects vehicles
-   Tracks object movement
-   Counts line crossings
-   Displays traffic level
-   Shows classical vs Grover comparison

##Output Metrics

Every second the system prints:

-   FPS
-   Capture latency
-   TPU inference time
-   Classical search time (64 evaluations)
-   Grover-style search time (~8 evaluations)
-   Speedup factor

Example output:

CLASSICAL (O(N)) evals=64 GROVER (~√N) evals=8 Speedup: ~7x

##How It Works

1.  Camera captures frame
2.  EdgeTPU runs object detection
3.  Scene encoded into 6-bit logical state (0–63)
4.  Classical search checks all 64 states
5.  Grover-style search checks ~8 states
6.  Results compared live

##Important Note

-   This project does NOT use quantum hardware.
-   It demonstrates a quantum-inspired decision search pattern
    implemented on embedded hardware.

##To Stop

-   Press ‘q’ in preview window
-   Press Ctrl + C in inference terminal
