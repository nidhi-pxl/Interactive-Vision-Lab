"""
Lightweight Test Script for Pinch Detection
Verifies PinchDetector state transitions (FSM), active hand selection,
and reset functionalities using mock landmark coordinates.
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.interaction.pinch_detector import PinchDetector


def make_mock_hand(thumb_tip: list[int], index_tip: list[int], hand_type: str = "Right") -> dict:
    """Helper to generate a mock hand dictionary containing required landmarks and hand type."""
    lm_list = [[0, 0, 0]] * 21
    lm_list[4] = [thumb_tip[0], thumb_tip[1], 0.0]
    lm_list[8] = [index_tip[0], index_tip[1], 0.0]
    return {"lm_list": lm_list, "type": hand_type}


def test_pinch_state_machine():
    print("Initializing PinchDetector with threshold=35.0...")
    detector = PinchDetector(threshold=35.0)

    # Initial state should be OPEN
    assert detector.state == "OPEN"
    assert not detector.pinching
    print("Initial state verification: PASSED")

    # Step 1: OPEN (Distance = 50.0 > 35.0)
    print("\nStep 1: Mocking OPEN frame (Distance = 50px)...")
    hand_open = make_mock_hand([100, 100], [100, 150])
    detector.update([hand_open], active_hand="first")
    print(f"Distance: {detector.distance:.1f}, Center: {detector.center}, State: {detector.state}, Pinching: {detector.pinching}")
    assert detector.distance == 50.0
    assert detector.center == (100, 125)
    assert detector.state == "OPEN"
    assert not detector.pinching
    print("Step 1 verification: PASSED")

    # Step 2: PINCH_STARTED (Distance = 20.0 < 35.0)
    print("\nStep 2: Mocking pinch start frame (Distance = 20px)...")
    hand_pinch_start = make_mock_hand([100, 100], [100, 120])
    detector.update([hand_pinch_start], active_hand="first")
    print(f"Distance: {detector.distance:.1f}, Center: {detector.center}, State: {detector.state}, Pinching: {detector.pinching}")
    assert detector.distance == 20.0
    assert detector.center == (100, 110)
    assert detector.state == "PINCH_STARTED"
    assert detector.pinching
    print("Step 2 verification: PASSED")

    # Step 3: PINCHING (Distance = 15.0 < 35.0)
    print("\nStep 3: Mocking holding pinch frame (Distance = 15px)...")
    hand_pinching = make_mock_hand([100, 100], [100, 115])
    detector.update([hand_pinching], active_hand="first")
    print(f"Distance: {detector.distance:.1f}, Center: {detector.center}, State: {detector.state}, Pinching: {detector.pinching}")
    assert detector.distance == 15.0
    assert detector.center == (100, 107)
    assert detector.state == "PINCHING"
    assert detector.pinching
    print("Step 3 verification: PASSED")

    # Step 4: PINCH_RELEASED (Distance = 60.0 > 35.0)
    print("\nStep 4: Mocking pinch release frame (Distance = 60px)...")
    hand_release = make_mock_hand([100, 100], [100, 160])
    detector.update([hand_release], active_hand="first")
    print(f"Distance: {detector.distance:.1f}, Center: {detector.center}, State: {detector.state}, Pinching: {detector.pinching}")
    assert detector.distance == 60.0
    assert detector.center == (100, 130)
    assert detector.state == "PINCH_RELEASED"
    assert not detector.pinching
    print("Step 4 verification: PASSED")

    # Step 5: OPEN (Distance = 70.0 > 35.0)
    print("\nStep 5: Mocking back to OPEN frame (Distance = 70px)...")
    hand_open_again = make_mock_hand([100, 100], [100, 170])
    detector.update([hand_open_again], active_hand="first")
    print(f"Distance: {detector.distance:.1f}, Center: {detector.center}, State: {detector.state}, Pinching: {detector.pinching}")
    assert detector.distance == 70.0
    assert detector.center == (100, 135)
    assert detector.state == "OPEN"
    assert not detector.pinching
    print("Step 5 verification: PASSED")


def test_active_hand_and_reset():
    print("\nTesting Active Hand selection and Reset logic...")
    detector = PinchDetector(threshold=35.0)

    # Initialize with a pinch state on the Right Hand
    hand_right_pinch = make_mock_hand([100, 100], [100, 120], hand_type="Right")
    hand_left_open = make_mock_hand([100, 100], [100, 180], hand_type="Left")

    # Active hand set to Right -> Should recognize pinch
    detector.update([hand_left_open, hand_right_pinch], active_hand="Right")
    assert detector.pinching
    assert detector.state == "PINCH_STARTED"

    # Active hand set to Left -> Should transition through PINCH_RELEASED (since Left is open)
    detector.update([hand_left_open, hand_right_pinch], active_hand="Left")
    assert not detector.pinching
    assert detector.state == "PINCH_RELEASED"

    # Next frame on Left -> Should be OPEN
    detector.update([hand_left_open, hand_right_pinch], active_hand="Left")
    assert not detector.pinching
    assert detector.state == "OPEN"

    # Reset test
    detector.update([hand_left_open, hand_right_pinch], active_hand="Right")  # pinch again
    assert detector.pinching
    detector.reset()
    assert detector.distance == 0.0
    assert detector.center == (0, 0)
    assert not detector.pinching
    assert detector.state == "OPEN"

    print("Active hand selection and Reset tests: PASSED")


def test_empty_hands():
    print("\nTesting empty hands list scenario...")
    detector = PinchDetector(threshold=35.0)
    
    # Update with some values first
    hand = make_mock_hand([100, 100], [100, 120])
    detector.update([hand], active_hand="first")
    assert detector.pinching
    
    # Update with empty list -> should automatically call reset
    detector.update([], active_hand="first")
    assert detector.distance == 0.0
    assert not detector.pinching
    assert detector.state == "OPEN"
    print("Empty hands list verification: PASSED")


if __name__ == "__main__":
    test_pinch_state_machine()
    test_active_hand_and_reset()
    test_empty_hands()
    print("\nAll PinchDetector unit tests completed successfully!")
