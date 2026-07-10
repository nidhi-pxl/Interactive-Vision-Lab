"""
Demo 2: Gesture & Finger Recognition System
Captures live webcam feed, processes hand landmarks using the reusable HandTracker,
extracts finger states, classifies hand gestures using the extensible GestureRecognizer,
and displays both finger count and high-level recognized gestures on a premium dashboard.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add the project root to sys.path to enable importing the shared src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.hand_tracker import HandTracker
from src.vision.gesture_recognizer import GestureRecognizer


def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be initialized.")
        return

    # Create the hand tracker and gesture recognizer instances
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    recognizer = GestureRecognizer()

    prev_time = 0.0

    print("Running Gesture & Finger Counter Demo. Press 'q' in the window to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        # Mirror the image horizontally for natural mirroring
        frame = cv2.flip(frame, 1)

        # Process the frame to detect hands
        tracker.process_frame(frame)

        # Visual overlay: Draw landmarks and connectivity lines
        frame = tracker.draw_landmarks(frame)

        # Retrieve structured hand data
        hands_info = tracker.get_hands_info(frame)

        total_fingers = 0
        hands_summary = []

        # Create a copy of the frame for drawing semi-transparent HUD elements
        hud_overlay = frame.copy()

        # Process each detected hand
        for hand in hands_info:
            bbox = hand["bbox"]
            hand_type = hand["type"]
            center = hand["center"]

            # Unpack bounding box
            xmin, ymin, w, h = bbox

            # Get fingers state list: [Thumb, Index, Middle, Ring, Pinky] (reusable representation)
            finger_states = tracker.get_finger_states(hand)
            raised_count = sum(finger_states)
            total_fingers += raised_count

            # Recognize the gesture using the GestureRecognizer mapping
            gesture_name = recognizer.recognize_gesture(finger_states, hand)

            # Store summary info for HUD display
            hands_summary.append({
                "type": hand_type,
                "count": raised_count,
                "gesture": gesture_name
            })

            # Draw a premium bounding box around the hand (soft green/cyan)
            cv2.rectangle(
                frame,
                (xmin - 10, ymin - 10),
                (xmin + w + 10, ymin + h + 10),
                (0, 255, 200),
                2,
            )

            # Draw a pill overlay at the hand center showing hand type and finger count
            cx, cy = center
            cv2.circle(frame, (cx, cy), 22, (0, 255, 200), cv2.FILLED)
            cv2.putText(
                frame,
                str(raised_count),
                (cx - 8, cy + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            # Display the recognized gesture name and handedness directly above the hand bounding box
            cv2.putText(
                frame,
                f"{hand_type} - {gesture_name}",
                (xmin - 10, ymin - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 200),
                2,
                cv2.LINE_AA,
            )

        # Draw a beautiful glassmorphism HUD panel in the top-left corner
        hud_w, hud_h = 300, 200
        hud_x, hud_y = 15, 15

        # Draw dark panel background
        cv2.rectangle(
            hud_overlay,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (25, 25, 25),
            cv2.FILLED,
        )
        
        # Blend the overlay with the main frame (alpha = 0.7)
        alpha = 0.7
        cv2.addWeighted(hud_overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Draw a subtle border around the HUD
        cv2.rectangle(
            frame,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (100, 100, 100),
            1,
        )

        # Overlay text inside the HUD
        cv2.putText(
            frame,
            "GESTURE RECOGNIZER",
            (hud_x + 15, hud_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Display the large total finger count
        cv2.putText(
            frame,
            f"TOTAL: {total_fingers}",
            (hud_x + 15, hud_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 200),
            3,
            cv2.LINE_AA,
        )

        # Display details for each active hand
        y_offset = 110
        for info in hands_summary:
            cv2.putText(
                frame,
                f"{info['type']}: {info['gesture']} ({info['count']}/5)",
                (hud_x + 15, hud_y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )
            y_offset += 25

        # Calculate and render FPS in the HUD
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (hud_x + 15, hud_y + 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        # Show the frame in a window
        cv2.imshow("Demo 2: Gesture & Finger Recognition", frame)

        # Graceful exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release webcam resources and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
