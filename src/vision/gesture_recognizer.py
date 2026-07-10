"""
Gesture Recognizer Module
Defines an extensible GestureRecognizer class that maps finger states
and landmark geometry to high-level gesture classifications.
"""


class GestureRecognizer:
    """
    Classifies hand gestures based on finger states and landmark coordinates.
    Supports easy template extensions.
    """

    def __init__(self):
        """
        Initializes default gesture templates.
        Templates are defined as a list of 5 elements matching:
        [Thumb, Index, Middle, Ring, Pinky]
        where 1 = open, 0 = closed, None = wildcard (either).
        """
        self.gestures = {
            "Open Palm": [1, 1, 1, 1, 1],
            "Fist": [0, 0, 0, 0, 0],
            "Pointing": [None, 1, 0, 0, 0],
            "Peace": [None, 1, 1, 0, 0],
            "Thumbs Up": [1, 0, 0, 0, 0],
        }

    def register_gesture(self, name: str, template: list) -> None:
        """
        Registers a new gesture template.

        Args:
            name: The name of the gesture.
            template: A list of 5 values (0, 1, or None) matching [Thumb, Index, Middle, Ring, Pinky].
        """
        if len(template) != 5:
            raise ValueError("Gesture template must contain exactly 5 elements.")
        self.gestures[name] = template

    def recognize_gesture(self, finger_states: list[int], hand_info: dict) -> str:
        """
        Classifies the gesture name based on finger states and hand landmark info.

        Args:
            finger_states: List of 5 states [Thumb, Index, Middle, Ring, Pinky].
            hand_info: Dictionary containing hand details (landmarks list, etc.).

        Returns:
            Recognized gesture name as a string, or "Unknown".
        """
        # Iterate over gesture templates to find matches
        for gesture_name, template in self.gestures.items():
            match = True
            for i in range(5):
                if template[i] is not None and template[i] != finger_states[i]:
                    match = False
                    break

            if match:
                # Custom geometric validation for specific gestures to improve reliability
                if gesture_name == "Thumbs Up":
                    # For a valid Thumbs Up, the thumb tip (4) must be higher (lower y)
                    # than the thumb MCP joint (2) and index knuckle (5).
                    lm_list = hand_info["lm_list"]
                    thumb_tip_y = lm_list[4][1]
                    thumb_mcp_y = lm_list[2][1]
                    index_mcp_y = lm_list[5][1]

                    if thumb_tip_y < thumb_mcp_y and thumb_tip_y < index_mcp_y:
                        return gesture_name
                    else:
                        continue  # Skip template match if orientation doesn't look like Thumbs Up

                return gesture_name

        return "Unknown"
