"""
gesture_flask.py — Web streaming gesture car
Raspberry Pi 5 + Pi Camera Module (CSI) + L298N + MediaPipe + Flask
Same logic as gesture_car-2.py, with Flask web streaming instead of cv2.imshow

MOTOR BEHAVIOR:
    FIST        -> Left FORWARD  + Right FORWARD   (car drives straight)
    OPEN PALM   -> Left STOP     + Right STOP      (car stops)
    THUMB LEFT  -> Left BACKWARD + Right FORWARD   (car spins LEFT in place)
    THUMB RIGHT -> Left FORWARD  + Right BACKWARD  (car spins RIGHT in place)
    anything else / no hand     -> STOP (default safe state)

L298N WIRING (BCM pin numbering):
    Pi GPIO 17 -> IN1     |  Controls LEFT motor direction
    Pi GPIO 27 -> IN2     |  Controls LEFT motor direction
    Pi GPIO 18 -> ENA     |  PWM for LEFT motor speed
    Pi GPIO 22 -> IN3     |  Controls RIGHT motor direction
    Pi GPIO 23 -> IN4     |  Controls RIGHT motor direction
    Pi GPIO 13 -> ENB     |  PWM for RIGHT motor speed
    Pi GND     -> L298N GND (tied to battery GND too — common ground is mandatory)

    L298N OUT1, OUT2 -> LEFT motor
    L298N OUT3, OUT4 -> RIGHT motor
    L298N 12V        -> battery +
    L298N GND        -> battery -
    L298N 5V-EN jumper: ON

STATUS LEDs:
    Pi GPIO 5  -> 330 ohm -> GREEN LED -> GND    (lit when motors running)
    Pi GPIO 6  -> 330 ohm -> RED   LED -> GND    (lit when motors stopped)

WEB ACCESS:
    Open http://<pi-ip>:5000 on any device on the same WiFi network
    Find Pi IP with: hostname -I
"""
import math
import cv2
import time
import threading
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from gpiozero import Motor, LED
import mediapipe as mp

# ─── MOTORS ──────────────────────────────────────────────────────
motor_left  = Motor(forward=17, backward=27, enable=18, pwm=True)
motor_right = Motor(forward=22, backward=23, enable=13, pwm=True)

SPEED      = 0.7
TURN_SPEED = 0.6

led_running = LED(5)
led_stopped = LED(6)

def set_status_leds(action):
    if action == "STOP":
        led_running.off()
        led_stopped.on()
    else:
        led_stopped.off()
        led_running.on()

def drive_forward():
    motor_left.forward(SPEED)
    motor_right.forward(SPEED)

def spin_left():
    motor_left.backward(TURN_SPEED)
    motor_right.forward(TURN_SPEED)

def spin_right():
    motor_left.forward(TURN_SPEED)
    motor_right.backward(TURN_SPEED)

def stop_motors():
    motor_left.stop()
    motor_right.stop()

ACTIONS = {
    "DRIVE": drive_forward,
    "LEFT":  spin_left,
    "RIGHT": spin_right,
    "STOP":  stop_motors,
}

# ─── MEDIAPIPE ───────────────────────────────────────────────────
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1,
)
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

# ─── CAMERA ──────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()
time.sleep(1)

# ─── SHARED STATE ────────────────────────────────────────────────
_frame_lock = threading.Lock()
_latest_frame = None
_capture_event = threading.Event()

_jpg_lock = threading.Lock()
_latest_jpg = None

_running = True

# ─── CAPTURE THREAD (fastest possible) ───────────────────────────
def capture_thread():
    global _latest_frame, _running
    while _running:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        with _frame_lock:
            _latest_frame = frame
        _capture_event.set()

# ─── DETECTION THREAD ────────────────────────────────────────────
def detection_thread():
    global _latest_jpg, _running
    last_gesture     = "NONE"
    stable_count     = 0
    STABLE_THRESHOLD = 5
    current_action   = "STOP"

    stop_motors()
    set_status_leds("STOP")

    while _running:
        _capture_event.wait(timeout=0.1)
        _capture_event.clear()

        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            continue

        frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        gesture = "NONE"

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            palm_size = math.sqrt(
                (lm[0].x - lm[9].x)**2 + (lm[0].y - lm[9].y)**2
            )
            thumb_dist  = math.sqrt((lm[4].x  - lm[5].x )**2 + (lm[4].y  - lm[5].y )**2)
            index_dist  = math.sqrt((lm[8].x  - lm[5].x )**2 + (lm[8].y  - lm[5].y )**2)
            middle_dist = math.sqrt((lm[12].x - lm[9].x )**2 + (lm[12].y - lm[9].y )**2)
            ring_dist   = math.sqrt((lm[16].x - lm[13].x)**2 + (lm[16].y - lm[13].y)**2)
            pinky_dist  = math.sqrt((lm[20].x - lm[17].x)**2 + (lm[20].y - lm[17].y)**2)

            thumb_up  = 1 if thumb_dist  > palm_size * 0.55  else 0
            index_up  = 1 if index_dist  > palm_size * 0.7  else 0
            middle_up = 1 if middle_dist > palm_size * 0.75 else 0
            ring_up   = 1 if ring_dist   > palm_size * 0.75 else 0
            pinky_up  = 1 if pinky_dist  > palm_size * 0.75 else 0
            other_fingers = middle_up + ring_up + pinky_up

            if thumb_up == 0 and index_up == 0 and other_fingers == 0:
                gesture = "FIST"
            elif index_up + middle_up + ring_up + pinky_up >= 3:
                gesture = "OPEN PALM"
            elif thumb_up == 1 and index_up == 0 and other_fingers == 0:
                dx = lm[4].x - lm[2].x
                dy = lm[4].y - lm[2].y
                if abs(dx) > abs(dy) * 0.8:
                    gesture = "RIGHT" if dx > 0 else "LEFT"
                else:
                    gesture = "UNKNOWN"
            else:
                gesture = "UNKNOWN"

            h, w, _ = frame.shape
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)),
                           5, (0, 255, 0), -1)
            color_map = {
                "RIGHT": (0, 165, 255), "LEFT": (0, 165, 255),
                "FIST": (0, 0, 255), "OPEN PALM": (255, 255, 255),
                "UNKNOWN": (100, 100, 100),
            }
            cv2.putText(frame, gesture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        color_map.get(gesture, (0, 255, 0)), 3)
        else:
            cv2.putText(frame, "No Hand", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if gesture == last_gesture:
            stable_count += 1
        else:
            stable_count = 0
            last_gesture = gesture

        if stable_count >= STABLE_THRESHOLD:
            target = {
                "FIST": "DRIVE", "OPEN PALM": "STOP",
                "LEFT": "LEFT", "RIGHT": "RIGHT",
            }.get(gesture, "STOP")
            if target != current_action:
                ACTIONS[target]()
                set_status_leds(target)
                current_action = target
                print(f"-> {gesture}: motors = {target}")

        cv2.putText(frame, f"motors: {current_action}",
                    (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        # Resize to half before encoding = 4x smaller JPEG = faster transfer
        small = cv2.resize(frame, (320, 240))
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ok:
            with _jpg_lock:
                _latest_jpg = buf.tobytes()

# ─── FLASK ───────────────────────────────────────────────────────
app = Flask(__name__)

PAGE = """
<!doctype html><html><head><title>Gesture Car</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{background:#111;color:#eee;text-align:center;font-family:sans-serif;margin:0;padding:10px}
img{max-width:100%;border:2px solid #444;border-radius:8px;image-rendering:auto}
h2{margin:8px 0}
</style>
</head><body>
<h2>Gesture Car — Live Feed</h2>
<img src="{{ url_for('video') }}">
</body></html>
"""

@app.route('/')
def index():
    return render_template_string(PAGE)

def gen_frames():
    while _running:
        with _jpg_lock:
            buf = _latest_jpg
        if buf is None:
            time.sleep(0.005)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ─── START ALL THREADS ───────────────────────────────────────────
print("\nStarting threads...")
threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=detection_thread, daemon=True).start()
threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False),
    daemon=True
).start()

print("Flask server: http://0.0.0.0:5000")
print("Find your Pi IP with: hostname -I")
print("Gesture car (web) running.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    _running = False
    print("\nStopping motors, turning off LEDs, cleaning up...")
    stop_motors()
    led_running.off()
    led_stopped.off()
    motor_left.close()
    motor_right.close()
    led_running.close()
    led_stopped.close()
    picam2.stop()
    picam2.close()