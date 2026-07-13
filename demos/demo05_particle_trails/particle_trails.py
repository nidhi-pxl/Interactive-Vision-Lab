"""
Demo 5: Interactive Particle Trails
Integrates HandTracker, PinchDetector, and the stateful ParticleSystem/Emitter graphics framework.
Emits physics-enabled particles (with gravity, drag, and alpha fade) on pinch.
Supports simulation controls: Pause (Space), Time Scaling (< >), size ([]), rate (+ -), and colors (1-6).
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add the project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.hand_tracker import HandTracker
from src.interaction.pinch_detector import PinchDetector
from src.graphics.particle_system import ParticleSystem
from src.graphics.particle_emitter import ParticleEmitter


def get_color_name(color: tuple[int, int, int], is_rainbow: bool) -> str:
    """Helper to return human-readable color name from BGR value."""
    if is_rainbow:
        return "Rainbow"
    if color == (0, 0, 255):
        return "Red"
    if color == (0, 255, 0):
        return "Green"
    if color == (255, 0, 0):
        return "Blue"
    if color == (0, 255, 255):
        return "Yellow"
    if color == (255, 255, 255):
        return "White"
    return "Custom"


def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be initialized.")
        return

    # Initialize tracker and detector (max hands = 2, first hand is active)
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = PinchDetector(threshold=35.0)

    # Initialize graphics: Particle System and Emitter
    # Default gravity pulls particles downwards (+y direction) at 250 px/s^2, drag slows them down
    system = ParticleSystem()
    emitter = ParticleEmitter(
        emission_rate=60.0,
        speed=120.0,
        spread=360.0,
        color=(0, 255, 0),
        radius=6.0,
        lifetime=1.5,
        gravity=(0.0, 250.0),
        drag=0.95,
    )

    prev_time = time.time()
    
    # Simulation speed configurations
    time_scales = [0.25, 0.5, 1.0, 2.0]
    time_scale_labels = ["0.25x", "0.5x", "1.0x", "2.0x"]
    time_scale_idx = 2  # Default to 1.0x
    paused = False

    # Rainbow mode trackers
    rainbow_mode = False
    rainbow_hue = 0

    print("Running Particle Trails Demo. Controls:")
    print("  - Pinch thumb and index to EMIT particles")
    print("  - Press 'Space' to Pause/Resume simulation")
    print("  - Press '<' (,) or '>' (.) to change time scale (slow-motion)")
    print("  - Press 'C' to clear particles")
    print("  - Press '[' or ']' to change particle radius")
    print("  - Press '+' or '-' to change emission rate")
    print("  - Press '1' to '6' to change color (6 = Rainbow Mode)")
    print("  - Press 'Q' to quit")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        # Mirror frame horizontally
        frame = cv2.flip(frame, 1)

        # Calculate time step (dt) in seconds
        curr_time = time.time()
        raw_dt = curr_time - prev_time
        prev_time = curr_time

        # Calculate scaled dt based on pause and time scale controls
        dt = 0.0
        if not paused:
            dt = raw_dt * time_scales[time_scale_idx]

        # Detect hands and draw landmarks
        tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame)
        hands_info = tracker.get_hands_info(frame)

        # Update pinch detector
        active_hand_config = "first"
        detector.update(hands_info, active_hand=active_hand_config)

        # Verify active hand is visible
        target_hand_found = False
        if hands_info:
            if active_hand_config == "first":
                target_hand_found = True
            else:
                for hand in hands_info:
                    if hand.get("type") == active_hand_config:
                        target_hand_found = True
                        break

        # Handle rainbow color cycling
        if rainbow_mode:
            rainbow_hue = (rainbow_hue + 3) % 180
            hsv_color = np.uint8([[[rainbow_hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            emitter.set_color((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))

        # Emit particles on active pinch state
        if target_hand_found:
            cx, cy = detector.center
            if detector.pinching and not paused:
                emitter.emit((cx, cy), system, dt)
        else:
            detector.reset()

        # Update active particle physics (time step)
        system.update(dt)

        # Draw particles onto camera frame
        system.draw(frame)

        # Draw brush cursor at pinch/hover center
        if target_hand_found:
            cx, cy = detector.center
            cursor_color = emitter.color if not rainbow_mode else (255, 255, 255)
            if detector.pinching:
                # Solid circle when pinching
                cv2.circle(frame, (cx, cy), int(emitter.radius), cursor_color, cv2.FILLED)
            else:
                # Outline circle when hovering
                cv2.circle(frame, (cx, cy), int(emitter.radius), (150, 150, 150), 1)

        # Render glassmorphic HUD status panel
        hud_overlay = frame.copy()
        hud_w, hud_h = 280, 190
        hud_x, hud_y = 15, 15
        
        cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (25, 25, 25), cv2.FILLED)
        alpha = 0.65
        cv2.addWeighted(hud_overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (100, 100, 100), 1)

        # Draw text stats inside HUD
        color_name = get_color_name(emitter.color, rainbow_mode)
        sim_speed = "PAUSED" if paused else time_scale_labels[time_scale_idx]
        
        cv2.putText(frame, "PARTICLE TRAILS", (hud_x + 15, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"PARTICLES: {len(system.particles)}", (hud_x + 15, hud_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Speed Scale: {sim_speed}", (hud_x + 15, hud_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Emission Rate: {int(emitter.emission_rate)}/s", (hud_x + 15, hud_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Size: {int(emitter.radius)}px | Color: {color_name}", (hud_x + 15, hud_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)

        # Raw FPS overlay (computed with raw_dt to show actual frame update rate)
        fps = 1.0 / raw_dt if raw_dt > 0.0 else 0.0
        cv2.putText(frame, f"FPS: {int(fps)}", (hud_x + 15, hud_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # Render window
        cv2.imshow("Demo 5: Stateful Particle Trails", frame)

        # Keyboard check
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            system.clear()
        elif key == ord(" "):
            paused = not paused
        elif key == ord(".") or key == ord(">"):  # Speed up
            time_scale_idx = min(len(time_scales) - 1, time_scale_idx + 1)
        elif key == ord(",") or key == ord("<"):  # Slow down
            time_scale_idx = max(0, time_scale_idx - 1)
        elif key == ord("]"):
            emitter.set_radius(emitter.radius + 1.0)
        elif key == ord("["):
            emitter.set_radius(emitter.radius - 1.0)
        elif key == ord("=") or key == ord("+"):
            emitter.set_emission_rate(emitter.emission_rate + 10.0)
        elif key == ord("-") or key == ord("_"):
            emitter.set_emission_rate(emitter.emission_rate - 10.0)
        elif key == ord("1"):
            rainbow_mode = False
            emitter.set_color((0, 0, 255))      # Red
        elif key == ord("2"):
            rainbow_mode = False
            emitter.set_color((0, 255, 0))      # Green
        elif key == ord("3"):
            rainbow_mode = False
            emitter.set_color((255, 0, 0))      # Blue
        elif key == ord("4"):
            rainbow_mode = False
            emitter.set_color((0, 255, 255))    # Yellow
        elif key == ord("5"):
            rainbow_mode = False
            emitter.set_color((255, 255, 255))  # White
        elif key == ord("6"):
            rainbow_mode = True                 # Rainbow Mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
