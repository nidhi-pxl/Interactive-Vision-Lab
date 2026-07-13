"""
Demo 5: Gesture Effects & Particle Trails
Integrates HandTracker, PinchDetector, GestureRecognizer, and the stateful ParticleSystem/Emitter.
Emits different dynamic particle effects based on the active hand gesture:
- Pinch: Customizable Particle Trail (Green by default, falls down, fully user-controlled).
- Open Palm: Fire Effect (Yellow-Orange-Red flames rising upwards).
- Pointing: Floating Magic Particles (Cyan-Blue-Purple particles hovering around fingertip).

Supports simulation controls (Pause, Time Scaling, and customization of the Pinch trail settings).
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
from src.vision.gesture_recognizer import GestureRecognizer
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

    # Initialize tracker, recognizer, and detector (max hands = 2, first hand is active)
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    recognizer = GestureRecognizer()
    detector = PinchDetector(threshold=35.0)

    # Initialize graphics: Particle System and Emitter
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

    # User-customized Pinch Trail settings (stored independently)
    trail_color = (0, 255, 0)
    trail_radius = 6.0
    trail_emission_rate = 60.0
    trail_rainbow_mode = False

    # Simulation speed configurations
    time_scales = [0.25, 0.5, 1.0, 2.0]
    time_scale_labels = ["0.25x", "0.5x", "1.0x", "2.0x"]
    time_scale_idx = 2  # Default to 1.0x
    paused = False

    # Rainbow mode trackers
    rainbow_hue = 0

    print("Running Gesture Effects Demo. Controls:")
    print("  - Pinch thumb and index to EMIT customizable Particle Trail")
    print("  - Open Palm to EMIT Fire Effect")
    print("  - Point index finger to EMIT Floating Magic Particles")
    print("  - Press 'Space' to Pause/Resume simulation")
    print("  - Press '<' (,) or '>' (.) to change time scale (slow-motion)")
    print("  - Press 'C' to clear particles")
    print("  - Press '[' or ']' to change pinch trail particle radius")
    print("  - Press '+' or '-' to change pinch trail emission rate")
    print("  - Press '1' to '6' to change pinch trail color (6 = Rainbow Mode)")
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
        target_hand = None
        if hands_info:
            if active_hand_config == "first":
                target_hand_found = True
                target_hand = hands_info[0]
            else:
                for hand in hands_info:
                    if hand.get("type") == active_hand_config:
                        target_hand_found = True
                        target_hand = hand
                        break

        active_gesture = "None"
        active_effect = "None"
        emitter_position = (0, 0)

        # Determine gesture behaviors and configure emitter
        if target_hand_found and target_hand is not None:
            # Get fingers state and recognize standard gesture
            finger_states = tracker.get_finger_states(target_hand)
            recognized_gesture = recognizer.recognize_gesture(finger_states, target_hand)

            # 1. Pinch Trail (highest priority when pinching)
            if detector.pinching:
                active_gesture = "Pinch"
                active_effect = "Pinch Trail"
                emitter_position = detector.center

                # Restore user's custom trail configurations
                # If rainbow mode is enabled, dynamically set color in cycle
                if trail_rainbow_mode:
                    rainbow_hue = (rainbow_hue + 3) % 180
                    hsv_color = np.uint8([[[rainbow_hue, 255, 255]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    current_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                else:
                    current_color = trail_color

                emitter.configure(
                    emission_rate=trail_emission_rate,
                    speed=120.0,
                    spread=360.0,
                    color=current_color,
                    radius=trail_radius,
                    lifetime=1.5,
                    gravity=(0.0, 250.0),
                    drag=0.95,
                    color_mode="normal",
                )

            # 2. Fire Effect (Open Palm)
            elif recognized_gesture == "Open Palm":
                active_gesture = "Open Palm"
                active_effect = "Fire Effect"
                # Use landmark 9 (middle MCP joint) as palm center
                lm9 = target_hand["lm_list"][9]
                emitter_position = (lm9[0], lm9[1])

                emitter.configure(
                    emission_rate=140.0,
                    speed=90.0,
                    spread=40.0,
                    color=(0, 255, 255),  # base color not used directly due to color_mode fire
                    radius=8.0,
                    lifetime=1.0,
                    gravity=(0.0, -180.0),  # Rise upwards
                    drag=0.98,
                    color_mode="fire",
                )

            # 3. Magic Floating Effect (Pointing)
            elif recognized_gesture == "Pointing":
                active_gesture = "Pointing"
                active_effect = "Magic Floating"
                # Use landmark 8 (index finger tip)
                lm8 = target_hand["lm_list"][8]
                emitter_position = (lm8[0], lm8[1])

                emitter.configure(
                    emission_rate=35.0,
                    speed=20.0,
                    spread=360.0,
                    color=(255, 255, 0),  # base color not used directly due to color_mode magic
                    radius=5.0,
                    lifetime=3.0,
                    gravity=(0.0, 0.0),  # Floating, no gravity
                    drag=0.99,  # High drag, slows down to hover
                    color_mode="magic",
                )
            else:
                active_gesture = recognized_gesture
                active_effect = "None"
        else:
            detector.reset()

        # Emit particles if an effect is active and not paused
        if target_hand_found and active_effect != "None" and not paused:
            emitter.emit(emitter_position, system, dt)

        # Update active particles physics
        system.update(dt)

        # Draw particles onto camera frame
        system.draw(frame)

        # Draw brush cursor overlays
        if target_hand_found:
            if active_effect == "Pinch Trail":
                cursor_color = emitter.color if not trail_rainbow_mode else (255, 255, 255)
                cv2.circle(frame, emitter_position, int(trail_radius), cursor_color, cv2.FILLED)
            elif active_effect == "Fire Effect":
                cv2.circle(frame, emitter_position, 10, (0, 140, 255), 2)  # Orange outline at palm
            elif active_effect == "Magic Floating":
                cv2.circle(frame, emitter_position, 6, (255, 0, 128), cv2.FILLED)  # Purple dot at tip
            else:
                # Normal hover circle at index-thumb midpoint when not emitting
                cv2.circle(frame, detector.center, int(trail_radius), (150, 150, 150), 1)

        # Render HUD status dashboard panel
        hud_overlay = frame.copy()
        hud_w, hud_h = 310, 215
        hud_x, hud_y = 15, 15
        
        cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (25, 25, 25), cv2.FILLED)
        alpha = 0.65
        cv2.addWeighted(hud_overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (100, 100, 100), 1)

        # Display texts inside HUD
        sim_speed = "PAUSED" if paused else time_scale_labels[time_scale_idx]
        trail_color_name = get_color_name(trail_color, trail_rainbow_mode)
        
        cv2.putText(frame, "GESTURE EFFECTS ENGINE", (hud_x + 15, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"EFFECT: {active_effect.upper()}", (hud_x + 15, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GESTURE: {active_gesture.upper()}", (hud_x + 15, hud_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"PARTICLES: {len(system.particles)}", (hud_x + 15, hud_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display trail properties
        cv2.putText(
            frame,
            f"Trail: {int(trail_emission_rate)}/s | Size: {int(trail_radius)}px | Color: {trail_color_name}",
            (hud_x + 15, hud_y + 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, f"Speed Scale: {sim_speed}", (hud_x + 15, hud_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # Calculate and overlay actual update FPS
        fps = 1.0 / raw_dt if raw_dt > 0.0 else 0.0
        cv2.putText(frame, f"FPS: {int(fps)}", (hud_x + 15, hud_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        # Render window
        cv2.imshow("Demo 5: Gesture Effects Engine", frame)

        # Keyboard checks
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
            trail_radius = min(50.0, trail_radius + 1.0)
        elif key == ord("["):
            trail_radius = max(2.0, trail_radius - 1.0)
        elif key == ord("=") or key == ord("+"):
            trail_emission_rate = min(500.0, trail_emission_rate + 10.0)
        elif key == ord("-") or key == ord("_"):
            trail_emission_rate = max(10.0, trail_emission_rate - 10.0)
        elif key == ord("1"):
            trail_rainbow_mode = False
            trail_color = (0, 0, 255)      # Red
        elif key == ord("2"):
            trail_rainbow_mode = False
            trail_color = (0, 255, 0)      # Green
        elif key == ord("3"):
            trail_rainbow_mode = False
            trail_color = (255, 0, 0)      # Blue
        elif key == ord("4"):
            trail_rainbow_mode = False
            trail_color = (0, 255, 255)    # Yellow
        elif key == ord("5"):
            trail_rainbow_mode = False
            trail_color = (255, 255, 255)  # White
        elif key == ord("6"):
            trail_rainbow_mode = True      # Rainbow Mode

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
