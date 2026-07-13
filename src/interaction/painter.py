"""
Painter Module
Provides a reusable, stateful Painter class for overlay-based drawing interaction.
Integrates drawing coordinates tracking and overlays without computer vision logic.

Design Documentation:
1. Persistent Canvas:
   The webcam feed is refreshed on every frame loop, meaning individual camera frames
   do not retain visual changes from frame to frame. A persistent drawing canvas
   (represented by an independent image buffer) acts as a persistent layer that
   accumulates and stores drawing modifications over time.
2. Previous-Point Tracking:
   Hand landmarks are sampled at discrete time intervals corresponding to the camera
   sampling rate (e.g. 30 FPS). If the hand moves rapidly, consecutive coordinate points
   will be separated by large pixel gaps. Tracking the previous coordinate allows drawing
   continuous lines (interpolating the path) between frames, preventing broken, dotted strokes.
3. FSM-Driven Stroke States:
   The PinchDetector state machine directly coordinates the drawing stroke lifecycle:
   - 'PINCH_STARTED': Triggers `begin_stroke`, setting up the initial tracking coordinate.
   - 'PINCHING': Triggers `continue_stroke`, drawing a line segment and updating the tracker.
   - 'PINCH_RELEASED': Triggers `end_stroke`, releasing the tracker to allow coordinate shifts
     without drawing lines.
"""

import cv2
import numpy as np


class Painter:
    """
    Manages a persistent drawing canvas layer and provides stroke lifecycle controls.
    """

    def __init__(self, brush_color: tuple[int, int, int] = (0, 255, 0), brush_size: int = 5):
        """
        Initializes drawing configurations and canvas parameters.

        Args:
            brush_color: BGR tuple representing stroke color.
            brush_size: Stroke thickness in pixels.
        """
        self.brush_color = brush_color
        self.brush_size = brush_size
        self.brush_style = "normal"  # Options: 'normal', 'marker', 'neon', 'spray', 'rainbow', 'eraser'
        self.canvas = None
        self.prev_point = None
        self.rainbow_hue = 0

    def set_color(self, color: tuple[int, int, int]) -> None:
        """
        Dynamically changes the brush stroke color.

        Args:
            color: BGR tuple representing the new brush color.
        """
        self.brush_color = color

    def set_size(self, size: int) -> None:
        """
        Dynamically changes the brush stroke thickness.

        Args:
            size: Stroke thickness in pixels.
        """
        self.brush_size = size

    def set_style(self, style: str) -> None:
        """
        Dynamically changes the brush style.

        Args:
            style: String matching 'normal', 'marker', 'neon', 'spray', 'rainbow', or 'eraser'.
        """
        valid_styles = ["normal", "marker", "neon", "spray", "rainbow", "eraser"]
        if style in valid_styles:
            self.brush_style = style
        else:
            raise ValueError(f"Invalid brush style: {style}. Must be one of {valid_styles}")

    def begin_stroke(self, position: tuple[int, int]) -> None:
        """
        Begins a new drawing stroke by initializing the previous point coordinate.

        Args:
            position: Coordinate (x, y) where the stroke starts.
        """
        self.prev_point = position

    def continue_stroke(self, position: tuple[int, int]) -> None:
        """
        Continues the drawing stroke on the canvas based on the active brush style.

        Args:
            position: Coordinate (x, y) representing the current brush position.
        """
        if self.canvas is None or self.prev_point is None:
            return

        import random
        import math

        # Determine brush color based on style
        if self.brush_style == "rainbow":
            # Cycle through Hues (0-179 in OpenCV HSV space)
            self.rainbow_hue = (self.rainbow_hue + 3) % 180
            hsv_color = np.uint8([[[self.rainbow_hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        elif self.brush_style == "eraser":
            color = (0, 0, 0)
        else:
            color = self.brush_color

        # Determine thickness
        thickness = self.brush_size
        if self.brush_style == "marker":
            thickness = int(self.brush_size * 2.0)
        elif self.brush_style == "eraser":
            thickness = int(self.brush_size * 2.5)

        # Draw on the canvas according to style
        if self.brush_style == "spray":
            # Generate random dots around the position
            radius = self.brush_size * 2
            # Interpolate spray points between previous position and current position for continuous spray
            steps = max(1, int(math.hypot(position[0] - self.prev_point[0], position[1] - self.prev_point[1]) / 5))
            for step in range(steps):
                t = step / steps
                ix = int(self.prev_point[0] + t * (position[0] - self.prev_point[0]))
                iy = int(self.prev_point[1] + t * (position[1] - self.prev_point[1]))

                # Draw a cluster of dots around this interpolated point
                for _ in range(self.brush_size * 2):
                    angle = random.uniform(0, 2 * math.pi)
                    dist = random.uniform(0, radius)
                    dx = int(dist * math.cos(angle))
                    dy = int(dist * math.sin(angle))
                    px = ix + dx
                    py = iy + dy

                    # Ensure coordinates are within canvas bounds
                    h, w, _ = self.canvas.shape
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(self.canvas, (px, py), 1, color, cv2.FILLED)
        elif self.brush_style == "neon":
            # Neon Style: wide soft glow line, then thin bright white core line
            cv2.line(self.canvas, self.prev_point, position, color, int(thickness * 2.5))
            cv2.line(self.canvas, self.prev_point, position, (255, 255, 255), int(thickness * 0.8))
        else:
            # Default linear drawing for normal, marker, rainbow, and eraser styles
            cv2.line(self.canvas, self.prev_point, position, color, thickness)

        # Update previous tracking point
        self.prev_point = position

    def end_stroke(self) -> None:
        """
        Terminates the current stroke and resets the tracking point.
        """
        self.prev_point = None

    def clear(self) -> None:
        """
        Clears all drawings from the canvas.
        """
        if self.canvas is not None:
            self.canvas.fill(0)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlays the drawing canvas onto the camera frame.
        Re-initializes canvas size lazily if frame dimensions change.

        Args:
            frame: Camera BGR frame image.

        Returns:
            The combined BGR image.
        """
        h, w, _ = frame.shape

        # Lazy initialization or resizing of canvas buffer
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Create a mask of the drawing (non-black pixels)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Blackout drawing pixels in the camera frame, and overlay colored drawings
        img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        img_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        return cv2.add(img_bg, img_fg)
