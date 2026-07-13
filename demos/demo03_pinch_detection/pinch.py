"""
Demo 3: Pinch Detection Application
Demonstrates real-time pinch interaction using the stateful PinchDetector.
Displays hand landmarks, finger tips connection line (changing color on pinch),
live distance, pinch states (OPEN / PINCH / PINCH_STARTED / PINCH_RELEASED), and FPS.
"""

import sys
import time
from pathlib import Path
import cv2

# Add the project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.hand_tracker import HandTracker
from src.interaction.pinch_detector import PinchDetector


def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be initialized.")
        return

    # Initialize tracker and detector (default threshold of 35 pixels)
    tracker = HandTracker(
        max_num_hands=2,  # Support tracking multiple hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = PinchDetector(threshold=35.0)

    prev_time = 0.0

    print("Running Pinch Detection Demo. Press 'q' in the window to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        # Mirror horizontally for natural feedback
        frame = cv2.flip(frame, 1)

        # Detect hand
        tracker.process_frame(frame)
        frame = tracker.draw_landmarks(frame)
        hands_info = tracker.get_hands_info(frame)

        # Select active hand configuration for the demo
        active_hand_config = "first"
        detector.update(hands_info, active_hand=active_hand_config)

        # Draw overlays if the active hand is found and tracked
        target_hand = None
        if hands_info:
            if active_hand_config == "first":
                target_hand = hands_info[0]
            else:
                for hand in hands_info:
                    if hand.get("type") == active_hand_config:
                        target_hand = hand
                        break

        if target_hand:
            # Extract tip coordinates for drawing
            x1, y1 = target_hand["lm_list"][4][0], target_hand["lm_list"][4][1]
            x2, y2 = target_hand["lm_list"][8][0], target_hand["lm_list"][8][1]
            cx, cy = detector.center

            # Determine line color: green for active pinch, magenta for open
            line_color = (0, 255, 0) if detector.pinching else (255, 0, 255)
            text_color = (0, 255, 0) if detector.pinching else (0, 165, 255)

            # Draw circles on the interaction landmarks: Thumb Tip (4) and Index Tip (8)
            cv2.circle(frame, (x1, y1), 8, line_color, cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, line_color, cv2.FILLED)

            # Draw connection line and pinch midpoint
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # Display live distance and FSM state near the pinch point
            cv2.putText(
                frame,
                f"{detector.state}",
                (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Dist: {int(detector.distance)}px",
                (cx + 15, cy + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Display simplified primary status text on-screen
            status_text = "PINCH" if detector.pinching else "OPEN"
            cv2.putText(
                frame,
                f"STATUS: {status_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                line_color,
                3,
                cv2.LINE_AA,
            )
        else:
            # If no active hand is detected, we show status and ensure detector is reset
            detector.reset()
            cv2.putText(
                frame,
                "STATUS: NO HAND",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Calculate and overlay FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        # Render window
        cv2.imshow("Demo 3: Stateful Pinch Detection", frame)

        # Quit cleanly on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
