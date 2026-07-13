"""
Particle Class Module
Defines a lightweight Particle data structure storing physical and visual state variables.
"""


class Particle:
    """
    Represents a single dynamic particle in the simulation.
    Stores position, velocity, sizing, lifetime, and custom physical factors (gravity/drag).
    """

    def __init__(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float,
        color: tuple[int, int, int],
        lifetime: float,
        gravity: tuple[float, float] = (0.0, 0.0),
        drag: float = 1.0,
        color_mode: str = "normal",
    ):
        """
        Initializes the state of a single particle.

        Args:
            x: Starting X coordinate.
            y: Starting Y coordinate.
            vx: Horizontal velocity component in pixels per second.
            vy: Vertical velocity component in pixels per second.
            radius: Initial size radius in pixels.
            color: BGR tuple representing the drawing color.
            lifetime: Duration in seconds the particle should exist.
            gravity: Acceleration vector (gx, gy) in pixels/sec^2.
            drag: Drag factor representing fraction of velocity retained per second.
            color_mode: Dynamic color behavior name ('normal', 'fire', 'magic').
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.age = 0.0
        self.gravity = gravity
        self.drag = drag
        self.color_mode = color_mode
