"""
Demo 1: Hand Tracking Application
Lightweight script handling camera initialization, the main application loop,
rendering results on screen, and clean shutdown.
All tracking logic is encapsulated in the reusable HandTracker class.
"""

import sys
import time
from pathlib import Path
import cv2

# Add the project root to sys.path to enable importing the shared src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.hand_tracker import HandTracker


def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be initialized.")
        return

    # Create the hand tracker instance
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    prev_time = 0.0

    print("Running Hand Tracking Demo. Press 'q' in the window to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read from webcam.")
            break

        # Mirror the image horizontally for a natural mirror-view experience
        frame = cv2.flip(frame, 1)

        # Process the frame to detect hands
        tracker.process_frame(frame)

        # Visual overlay: Draw landmarks and connectivity lines
        frame = tracker.draw_landmarks(frame)

        # Retrieve structured hand data
        hands_info = tracker.get_hands_info(frame)

        # Draw bounding boxes, handedness, and confidence scores
        for hand in hands_info:
            bbox = hand["bbox"]
            hand_type = hand["type"]
            score = hand["score"]

            # Unpack bounding box
            xmin, ymin, w, h = bbox

            # Draw a green bounding rectangle around each detected hand
            cv2.rectangle(
                frame,
                (xmin - 15, ymin - 15),
                (xmin + w + 15, ymin + h + 15),
                (0, 255, 0),
                2,
            )

            # Label the hand with handedness (Left/Right) and confidence score
            cv2.putText(
                frame,
                f"{hand_type} ({score:.2f})",
                (xmin - 10, ymin - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Calculate and render FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

        # Show the frame in a window
        cv2.imshow("Demo 1: Hand Tracking", frame)

        # Graceful exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release webcam resources and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
