"""
Demo 4: Interactive Hand Painter
Integrates HandTracker, PinchDetector, and the reusable Painter module.
Draws continuous lines when pinching, allows runtime controls (C to clear, S to save as PNG, Q to quit),
and displays landmarks, cursor size, active FSM state, and FPS.
"""

import sys
import time
import datetime
from pathlib import Path
import cv2

# Add the project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.hand_tracker import HandTracker
from src.interaction.pinch_detector import PinchDetector
from src.interaction.painter import Painter


def get_color_name(color: tuple[int, int, int], style: str) -> str:
    """Helper to return human-readable color name from BGR value."""
    if style == "eraser":
        return "Eraser"
    if style == "rainbow":
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

    # Initialize vision, interaction, and graphics layers
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = PinchDetector(threshold=35.0)
    painter = Painter(brush_color=(0, 255, 0), brush_size=8)

    prev_time = 0.0
    save_notification_timer = 0
    save_notification_text = ""

    print("Running Hand Painter. Controls:")
    print("  - Pinch thumb and index to DRAW")
    print("  - Press 'C' to clear canvas")
    print("  - Press 'S' to save painting as PNG")
    print("  - Press 'Q' to quit")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        # Mirror frame horizontally
        frame = cv2.flip(frame, 1)

        # Detect hands and draw landmarks
        tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame)
        hands_info = tracker.get_hands_info(frame)

        # Update pinch state selector using primary hand
        active_hand_config = "first"
        detector.update(hands_info, active_hand=active_hand_config)

        # Check for target hand availability
        target_hand_found = False
        if hands_info:
            if active_hand_config == "first":
                target_hand_found = True
            else:
                for hand in hands_info:
                    if hand.get("type") == active_hand_config:
                        target_hand_found = True
                        break

        # Apply FSM-driven drawing actions
        if target_hand_found:
            cx, cy = detector.center

            if detector.state == "PINCH_STARTED":
                painter.begin_stroke((cx, cy))
            elif detector.state == "PINCHING":
                painter.continue_stroke((cx, cy))
            elif detector.state == "PINCH_RELEASED":
                painter.end_stroke()
        else:
            detector.reset()
            painter.end_stroke()

        # Render the drawing canvas onto the webcam frame
        frame = painter.render(frame)

        # Draw the brush cursor
        if target_hand_found:
            cx, cy = detector.center
            if detector.pinching:
                # Solid filled circle in active brush color when drawing
                cv2.circle(frame, (cx, cy), painter.brush_size, painter.brush_color, cv2.FILLED)
            else:
                # Semi-transparent/dotted outline showing brush cursor position before drawing
                cv2.circle(frame, (cx, cy), painter.brush_size, (150, 150, 150), 1)

        # Draw overlays: Status and active brush settings
        status_text = "DRAWING" if detector.pinching else "HOVERING"
        cv2.putText(
            frame,
            f"STATE: {detector.state} ({status_text})",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            painter.brush_color if detector.pinching else (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        color_name = get_color_name(painter.brush_color, painter.brush_style)
        cv2.putText(
            frame,
            f"BRUSH: {color_name} | STYLE: {painter.brush_style.upper()} | SIZE: {painter.brush_size}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "1-5: Color | 6: Eraser | N: Normal | M: Marker | P: Spray | R: Rainbow | O: Neon | []: Size",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Calculate and overlay FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        # Show saving notification overlay if active
        if save_notification_timer > 0:
            cv2.putText(
                frame,
                save_notification_text,
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            save_notification_timer -= 1

        # Render frame
        cv2.imshow("Demo 4: Interactive Hand Painter", frame)

        # Keyboard checks
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            painter.clear()
            save_notification_text = "Canvas Cleared!"
            save_notification_timer = 45  # Show for 45 frames (~1.5s)
        elif key == ord("s"):
            if painter.canvas is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"painting_{timestamp}.png"
                cv2.imwrite(filename, painter.canvas)
                save_notification_text = f"Saved: {filename}"
                save_notification_timer = 90  # Show for 90 frames (~3.0s)
                print(f"Canvas saved successfully as: {filename}")
        elif key == ord("1"):
            painter.set_color((0, 0, 255))      # Red (BGR)
            if painter.brush_style == "eraser":
                painter.set_style("normal")
        elif key == ord("2"):
            painter.set_color((0, 255, 0))      # Green (BGR)
            if painter.brush_style == "eraser":
                painter.set_style("normal")
        elif key == ord("3"):
            painter.set_color((255, 0, 0))      # Blue (BGR)
            if painter.brush_style == "eraser":
                painter.set_style("normal")
        elif key == ord("4"):
            painter.set_color((0, 255, 255))    # Yellow (BGR)
            if painter.brush_style == "eraser":
                painter.set_style("normal")
        elif key == ord("5"):
            painter.set_color((255, 255, 255))  # White (BGR)
            if painter.brush_style == "eraser":
                painter.set_style("normal")
        elif key == ord("6"):
            painter.set_style("eraser")
        elif key == ord("n"):
            painter.set_style("normal")
        elif key == ord("m"):
            painter.set_style("marker")
        elif key == ord("p"):
            painter.set_style("spray")
        elif key == ord("r"):
            painter.set_style("rainbow")
        elif key == ord("o"):
            painter.set_style("neon")
        elif key == ord("]"):
            painter.set_size(min(50, painter.brush_size + 2))
        elif key == ord("["):
            painter.set_size(max(2, painter.brush_size - 2))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
