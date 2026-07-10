"""
Lightweight Test Script for Gesture Classification
Verifies GestureRecognizer template matching and geometric orientation filtering
against mock landmark data.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable importing the shared src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.vision.gesture_recognizer import GestureRecognizer


def test_gesture_recognizer():
    print("Initializing GestureRecognizer...")
    recognizer = GestureRecognizer()

    # Mock landmark data for hand orientation validation (e.g. Thumbs Up check)
    # Joint IDs: Wrist (0), Thumb MCP (2), Thumb Tip (4), Index MCP (5)
    # Scenario A: Thumb pointing upward (tip y < mcp y) -> Valid Thumbs Up
    hand_info_up = {
        "lm_list": [
            [0, 100, 0],  # 0: Wrist
            [0, 0, 0],    # 1
            [40, 50, 0],  # 2: Thumb MCP (y = 50)
            [0, 0, 0],    # 3
            [40, 20, 0],  # 4: Thumb Tip (y = 20, higher/lower y than MCP/Index MCP)
            [30, 45, 0],  # 5: Index MCP (y = 45)
            # Other landmarks filled with zeros
            *[[0, 0, 0]] * 15,
        ]
    }

    # Scenario B: Thumb pointing downward (tip y > mcp y) -> Invalid Thumbs Up
    hand_info_down = {
        "lm_list": [
            [0, 100, 0],  # 0: Wrist
            [0, 0, 0],    # 1
            [40, 50, 0],  # 2: Thumb MCP (y = 50)
            [0, 0, 0],    # 3
            [40, 80, 0],  # 4: Thumb Tip (y = 80, lower/higher y than MCP)
            [30, 45, 0],  # 5: Index MCP (y = 45)
            *[[0, 0, 0]] * 15,
        ]
    }

    # 1. Test Open Palm: [1, 1, 1, 1, 1]
    gesture = recognizer.recognize_gesture([1, 1, 1, 1, 1], hand_info_up)
    print(f"Template [1, 1, 1, 1, 1] -> Recognized: {gesture}")
    assert gesture == "Open Palm", f"Expected Open Palm, got {gesture}"

    # 2. Test Fist: [0, 0, 0, 0, 0]
    gesture = recognizer.recognize_gesture([0, 0, 0, 0, 0], hand_info_up)
    print(f"Template [0, 0, 0, 0, 0] -> Recognized: {gesture}")
    assert gesture == "Fist", f"Expected Fist, got {gesture}"

    # 3. Test Pointing: [0, 1, 0, 0, 0]
    gesture = recognizer.recognize_gesture([0, 1, 0, 0, 0], hand_info_up)
    print(f"Template [0, 1, 0, 0, 0] -> Recognized: {gesture}")
    assert gesture == "Pointing", f"Expected Pointing, got {gesture}"

    # 4. Test Peace: [0, 1, 1, 0, 0]
    gesture = recognizer.recognize_gesture([0, 1, 1, 0, 0], hand_info_up)
    print(f"Template [0, 1, 1, 0, 0] -> Recognized: {gesture}")
    assert gesture == "Peace", f"Expected Peace, got {gesture}"

    # 5. Test Thumbs Up (Valid orientation)
    gesture = recognizer.recognize_gesture([1, 0, 0, 0, 0], hand_info_up)
    print(f"Template [1, 0, 0, 0, 0] (Pointing Up) -> Recognized: {gesture}")
    assert gesture == "Thumbs Up", f"Expected Thumbs Up, got {gesture}"

    # 6. Test Thumbs Up (Invalid orientation)
    gesture = recognizer.recognize_gesture([1, 0, 0, 0, 0], hand_info_down)
    print(f"Template [1, 0, 0, 0, 0] (Pointing Down) -> Recognized: {gesture}")
    assert gesture == "Unknown", f"Expected Unknown, got {gesture}"

    print("All gesture classification tests passed successfully!")


if __name__ == "__main__":
    test_gesture_recognizer()
