# Gesture-Controlled Toy Car — Edge AI Project

A real-time, contactless vehicle control system that recognizes hand gestures through a camera feed and translates them into motor commands — running entirely on a **Raspberry Pi 5** with zero cloud dependency.

**Course:** Edge AI  
**Student:** D Rushikesh (26501) Rudrani Barik (26161) Aman Kumar Rai (26750) Prajwal GM (26654)
**Supervisor:** Prof. Pandarasamy

---

## Demo

| Gesture | Motor Action | LED Status |
|---------|-------------|------------|
| ✊ Fist | Drive Forward | Green ON |
| 🖐 Open Palm | Stop | Red ON |
| 👈 Thumb Left | Spin Left in place | Green ON |
| 👉 Thumb Right | Spin Right in place | Green ON |
| No Hand / Unknown | Stop (safe default) | Red ON |

### Screenshots

**FIST → DRIVE (with CPU/GPU stats)**
![FIST](images/screenshots/CPU_FIST.png)

**OPEN PALM → STOP**
![OPEN PALM](images/screenshots/CPU_OPEN_PALM.png)

**No Hand → STOP**
![NO HAND](images/screenshots/CPU_NO_HAND.png)

**LEFT → Spin Left**
![LEFT](images/screenshots/LEFT.png)

**RIGHT → Spin Right**
![RIGHT](images/screenshots/RIGHT.png)

**Assembled Car**
![Car](images/hardware/car_assembled.jpeg)

---

## How It Works

```
Pi Camera (640×480 RGB, 30 fps)
    │
    ▼
MediaPipe Hand Landmarker (TFLite + XNNPACK, on-device CPU)
    │
    ▼
21 Hand Landmarks (x, y, z)
    │
    ▼
Geometric Gesture Classifier (rule-based, no training needed)
    │
    ▼
3-Frame Debounce Filter (prevents motor jitter)
    │
    ▼
GPIO → L298N Motor Driver → Left/Right DC Motors
    │
    ▼
Status LEDs (Green = moving, Red = stopped)
```

The gesture classifier computes finger-tip-to-base distances relative to palm size. No custom dataset or model training is required — classification is purely geometric using MediaPipe's pre-trained landmark output.

**Detection range:** up to ~3 meters from camera.

---

## Project Structure

```
gesture-car-edge-ai/
├── README.md                  ← You are here
├── report.md                  ← Full project report (Edge AI course format)
├── requirements.txt           ← Python dependencies
├── gesture_car.py           ← Main control script (the only code file)
├── hand_landmarker.task       ← MediaPipe pre-trained model (TFLite)
│
└── images/
    ├── hardware/
    │   ├── car_chassis.jpeg       ← 2WD chassis with DC motors
    │   ├── car_assembled.jpeg     ← Fully assembled car
    │   ├── l298n_wiring.jpeg      ← L298N close-up with GPIO wires
    │   ├── pi_camera_v2.png       ← Pi Camera Module V2
    │   ├── rpi_pin.jpg            ← RPi 5 GPIO pinout reference
    │   └── l298n_pin.png          ← L298N module pinout reference
    │
    └── screenshots/
        ├── CPU_FIST.png           ← FIST detected + CPU/GPU/RAM stats
        ├── CPU_NO_HAND.png        ← No hand + system stats
        ├── CPU_OPEN_PALM.png      ← Open palm + system stats
        ├── CPU_OPEN_PALM_1.png    ← Open palm (alternate angle)
        ├── DRIVE.png              ← Terminal: FIST → DRIVE
        ├── OPEN.png               ← Terminal: OPEN PALM → STOP
        ├── LEFT.png               ← Terminal: LEFT → spin LEFT
        ├── RIGHT.png              ← Terminal: RIGHT → spin RIGHT
        ├── NO_HAND.png            ← Terminal: No Hand → STOP
        └── UNKNOWN.png            ← Terminal: UNKNOWN → STOP
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Raspberry Pi 5 | 16 GB RAM, ARM Cortex-A76 quad-core |
| Pi Camera Module V2 | 8 MP Sony IMX219, CSI ribbon cable |
| L298N Dual H-Bridge | Motor driver module |
| DC Gear Motors × 2 | 3–6V yellow TT motors |
| 2WD Car Chassis | Acrylic platform + rear caster wheel |
| Battery Pack | 7.4V Li-ion (2S) or equivalent |
| Jumper Wires | Male-to-female, Male-to-male, Female-to-female assorted colors |

---

## Wiring (BCM Pin Numbering)

```
Raspberry Pi 5                L298N Motor Driver
─────────────                ──────────────────
GPIO 17  ──────────────────► IN1   (Left motor direction A)
GPIO 27  ──────────────────► IN2   (Left motor direction B)
GPIO 18  ──────────────────► ENA   (Left motor PWM speed)
GPIO 22  ──────────────────► IN3   (Right motor direction A)
GPIO 23  ──────────────────► IN4   (Right motor direction B)
GPIO 13  ──────────────────► ENB   (Right motor PWM speed)
GND      ──────────────────► GND   (⚠ common ground — MANDATORY)

L298N OUT1, OUT2  ──►  Left DC Motor
L298N OUT3, OUT4  ──►  Right DC Motor
L298N 12V         ──►  Battery +
L298N GND         ──►  Battery −
L298N 5V-EN jumper: ON (use on-board regulator)
```

> **Critical:** Pi GND, L298N GND, and Battery GND must all be connected together. Without a common ground, GPIO signals won't be read correctly by the L298N.

**Wiring reference photos:**

| RPi 5 GPIO Pinout | L298N Pinout |
|-------------------|-------------|
| ![RPi Pins](images/hardware/rpi_pin.jpg) | ![L298N Pins](images/hardware/l298n_pin.png) |

---

## Step-by-Step Reproduction

### 1. Assemble the Hardware

1. Mount two DC gear motors on the 2WD chassis and attach wheels.
2. Attach caster wheel at the rear for stability.
3. Secure the Raspberry Pi 5 on top of the chassis.
4. Mount the L298N motor driver beside the Pi.
5. Connect the Pi Camera Module V2 via CSI ribbon cable.
6. Wire GPIO pins to L298N following the wiring table above.
8. Connect battery pack to L298N 12V and GND terminals.
9. Connect a jumper wire from Pi GND to L298N GND (common ground).

### 2. Set Up the Software

```bash
# Update the Pi
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-venv python3-pip libcamera-apps

# Clone this repository
git clone https://github.com/RBarik25/gesture_control_toy_car.git
cd gesture_control_toy_car

# Create and activate virtual environment
python3 -m venv rpi-cv-env
source rpi-cv-env/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the MediaPipe Model (if not included)

The `hand_landmarker.task` file is included in this repo. If missing, download it:

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 4. Run

```bash
source rpi-cv-env/bin/activate
python3 gesture_car.py
```

**Expected terminal output:**
```
Gesture car running.
  FIST        -> FORWARD    (green LED on)
  OPEN PALM   -> STOP       (red LED on)
  THUMB LEFT  -> spin LEFT  (green LED on)
  THUMB RIGHT -> spin RIGHT (green LED on)
Press 'q' to quit.

→ FIST: motors = DRIVE
→ OPEN PALM: motors = STOP
→ LEFT: motors = LEFT
→ RIGHT: motors = RIGHT
```

An OpenCV window named **"Gesture Car"** opens showing the live camera feed with green landmark dots, gesture label, and motor state overlay.

**To quit:** Press `q` on the OpenCV window. Motors stop, LEDs turn off, camera closes cleanly.

---

## On-Device Performance (Raspberry Pi 5)

Measured from Task Manager while running `gesture_car.py`:

| Metric | Value |
|--------|-------|
| CPU usage (system) | 25–29% |
| CPU usage (python3 process) | 24–26% |
| GPU usage | 17–24% (display compositor only) |
| RAM usage | ~1.5 GB of 16 GB (~9.5%) |
| python3 VM size | ~3.0 GB |
| Frame resolution | 640 × 480 |
| Detection range | ~3 meters |
| Motor response latency | ~100 ms (3-frame debounce) |

The system leaves ~70% CPU and ~14 GB RAM free, confirming the RPi 5 handles this workload comfortably.

---

## Gesture Classification Logic

The classifier uses **no machine learning** — it's purely geometric, computed from MediaPipe's 21 landmarks:

1. **Palm size** = distance(wrist → middle finger MCP) — used as scale reference.
2. **Finger extension** = tip-to-base distance / palm size, per finger.
3. **Thresholds:** thumb > 0.5, index > 0.65, middle/ring/pinky > 0.7.
4. **FIST:** all fingers below threshold.
5. **OPEN PALM:** ≥ 3 of 4 non-thumb fingers above threshold.
6. **THUMB LEFT/RIGHT:** only thumb extended, horizontal displacement checked.
7. **Debounce:** 3 consecutive identical classifications required before motor command fires.

---

## Presentation and demo Link:

https://drive.google.com/file/d/1SGjZ2tPcxkoMyVv4deopUdl5ZfRIwkbu/view?usp=sharing

https://drive.google.com/file/d/1eOKqtcF5Vrtj3E8GR643PfRDqlS4vYUe/view?usp=drivesdk

---

## Dependencies

```
mediapipe       — Hand landmark detection (TFLite model)
opencv-python   — Frame capture, display, drawing
picamera2       — CSI camera interface (libcamera backend)
gpiozero        — GPIO control for motors and LEDs
```

All run on **Python 3.11** within a virtual environment on Raspberry Pi OS (64-bit, Bookworm).

---

## External Resources Used

| Resource | Link |
|----------|------|
| MediaPipe Hand Landmarker model | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task |
| MediaPipe documentation | https://developers.google.com/mediapipe/solutions/vision/hand_landmarker |
| RPi 5 GPIO docs | https://www.raspberrypi.com/documentation/computers/raspberry-pi.html |
| Picamera2 manual | https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf |
| gpiozero docs | https://gpiozero.readthedocs.io/ |
| L298N datasheet | https://www.st.com/resource/en/datasheet/l298.pdf |

No external datasets were used. The gesture classification is rule-based and requires no training data.

---

## License

This project was developed for academic purposes as part of the Edge AI course, IISc, Bengaluru

Course Website : https://www.samy101.com/edge-ai-26/
