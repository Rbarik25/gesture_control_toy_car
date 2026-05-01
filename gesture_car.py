"""
Gesture-controlled car — Raspberry Pi 5 + Pi Camera Module (CSI) + L298N + MediaPipe
                        + status LEDs (green = running, red = stopped)

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
"""

import math
import cv2
import mediapipe as mp
from gpiozero import Motor, LED
from picamera2 import Picamera2

# ─── MOTORS via L298N ─────────────────────────────────────────────
motor_left  = Motor(forward=17, backward=27, enable=18, pwm=True)   # IN1, IN2, ENA
motor_right = Motor(forward=22, backward=23, enable=13, pwm=True)   # IN3, IN4, ENB

SPEED      = 0.7
TURN_SPEED = 0.6

# ─── STATUS LEDs ──────────────────────────────────────────────────
led_running = LED(5)   # green — motors active
led_stopped = LED(6)   # red   — motors stopped


def set_status_leds(action):
    """Light green if moving, red if stopped."""
    if action == "STOP":
        led_running.off()
        led_stopped.on()
    else:   # DRIVE, LEFT, RIGHT
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

# ─── MEDIAPIPE HAND LANDMARKER ────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

# ─── CAMERA (Pi Camera Module via CSI ribbon) ─────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
picam2.start()

# ─── STATE ────────────────────────────────────────────────────────
last_gesture     = "NONE"
stable_count     = 0
STABLE_THRESHOLD = 3
current_action   = "STOP"

# Initial state: motors off, red LED on
stop_motors()
set_status_leds("STOP")

print("Gesture car running.")
print("  FIST        -> FORWARD    (green LED on)")
print("  OPEN PALM   -> STOP       (red LED on)")
print("  THUMB LEFT  -> spin LEFT  (green LED on)")
print("  THUMB RIGHT -> spin RIGHT (green LED on)")
print("Press 'q' to quit.\n")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
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

            thumb_up  = 1 if thumb_dist  > palm_size * 0.5  else 0
            index_up  = 1 if index_dist  > palm_size * 0.65 else 0
            middle_up = 1 if middle_dist > palm_size * 0.7  else 0
            ring_up   = 1 if ring_dist   > palm_size * 0.7  else 0
            pinky_up  = 1 if pinky_dist  > palm_size * 0.7  else 0

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

            # Draw landmarks
            h, w, _ = frame.shape
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)),
                           5, (0, 255, 0), -1)

            color_map = {
                "RIGHT": (0, 165, 255), "LEFT": (0, 165, 255),
                "FIST": (0, 0, 255),    "OPEN PALM": (255, 255, 255),
                "UNKNOWN": (100, 100, 100),
            }
            cv2.putText(frame, gesture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        color_map.get(gesture, (0, 255, 0)), 3)
        else:
            cv2.putText(frame, "No Hand", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # ─── Debounce + default-to-STOP ──────────────────────────
        if gesture == last_gesture:
            stable_count += 1
        else:
            stable_count = 0
            last_gesture = gesture

        if stable_count >= STABLE_THRESHOLD:
            target = {
                "FIST":      "DRIVE",
                "OPEN PALM": "STOP",
                "LEFT":      "LEFT",
                "RIGHT":     "RIGHT",
            }.get(gesture, "STOP")

            if target != current_action:
                ACTIONS[target]()
                set_status_leds(target)          # ← update LEDs
                current_action = target
                print(f"→ {gesture}: motors = {target}")

        cv2.putText(frame, f"motors: {current_action}",
                    (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.imshow("Gesture Car", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
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
    cv2.destroyAllWindows()