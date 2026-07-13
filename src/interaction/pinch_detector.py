"""
Pinch Detector Module
Provides a stateful PinchDetector class that tracks the geometric pinch interaction
state of a hand without performing direct computer vision or MediaPipe processing.
"""

import math


class PinchDetector:
    """
    Tracks and classifies the pinch interaction state using hand landmark data.

    A pinch gesture is defined by the proximity of the thumb tip (landmark 4)
    and the index finger tip (landmark 8). These two fingers represent the primary
    opposable digits in human hand anatomy used for fine selection and manipulation.
    Euclidean distance is used to measure proximity, serving as a reliable analogue
    to a physical button press or click.

    Exposes read-only interaction properties:
    - distance: Euclidean distance in pixels between thumb and index tips.
    - center: Center coordinates (cx, cy) between thumb and index tips.
    - pinching: Boolean flag indicating if the hand is currently pinching.
    - state: FSM state string ('OPEN', 'PINCH_STARTED', 'PINCHING', 'PINCH_RELEASED').
    """

    def __init__(self, threshold: float = 35.0):
        """
        Initializes the PinchDetector with a configurable distance threshold.

        Args:
            threshold: Distance in pixels below which a pinch is detected.
        """
        self.threshold = threshold

        # Stateful interaction properties
        self._distance = 0.0
        self._center = (0, 0)
        self._pinching = False
        self._state = "OPEN"

    @property
    def distance(self) -> float:
        """Returns the current Euclidean distance between thumb and index tips."""
        return self._distance

    @property
    def center(self) -> tuple[int, int]:
        """Returns the current midpoint coordinates between thumb and index tips."""
        return self._center

    @property
    def pinching(self) -> bool:
        """Returns True if the detector is currently in a pinch state."""
        return self._pinching

    @property
    def state(self) -> str:
        """
        Returns the current state machine state:
        - 'OPEN': No pinch active.
        - 'PINCH_STARTED': Just entered pinch state (exactly one frame).
        - 'PINCHING': Retaining active pinch state.
        - 'PINCH_RELEASED': Just exited pinch state (exactly one frame).
        """
        return self._state

    def reset(self) -> None:
        """Resets all interaction states to their default open values."""
        self._distance = 0.0
        self._center = (0, 0)
        self._pinching = False
        self._state = "OPEN"

    def update(self, hands_info: list[dict], active_hand: str = "first") -> None:
        """
        Selects the active hand and updates the pinch interaction state.

        Args:
            hands_info: List of hand dictionaries returned by HandTracker.get_hands_info().
            active_hand: Specifies which hand to track: 'first', 'Left', or 'Right'.
        """
        if not hands_info:
            self.reset()
            return

        # Select the target hand based on configuration
        target_hand = None
        if active_hand == "first":
            target_hand = hands_info[0]
        else:
            for hand in hands_info:
                if hand.get("type") == active_hand:
                    target_hand = hand
                    break

        if target_hand is None:
            self.reset()
            return

        lm_list = target_hand.get("lm_list", [])
        if len(lm_list) < 21:
            self.reset()
            return

        # Extract coordinates for Thumb Tip (4) and Index Tip (8)
        # lm format is [x, y, z]
        x1, y1 = lm_list[4][0], lm_list[4][1]
        x2, y2 = lm_list[8][0], lm_list[8][1]

        # Calculate Euclidean distance (2D)
        self._distance = math.hypot(x2 - x1, y2 - y1)

        # Calculate pinch center (midpoint)
        self._center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Determine current pinch threshold crossing
        is_currently_pinching = self._distance < self.threshold

        # Update the Finite State Machine (FSM) transitions
        if is_currently_pinching:
            if self._pinching:
                self._state = "PINCHING"
            else:
                self._state = "PINCH_STARTED"
                self._pinching = True
        else:
            if self._pinching:
                self._state = "PINCH_RELEASED"
                self._pinching = False
            else:
                self._state = "OPEN"
