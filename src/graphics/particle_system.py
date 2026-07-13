"""
Particle System Module
Manages a collection of active particles, updates their physics (velocity, gravity, drag),
and handles rendering (color fading and radius shrinking over lifetime).
"""

import math
import cv2
import numpy as np
from src.graphics.particle import Particle


class ParticleSystem:
    """
    Manages active particles, updates their states, and draws them onto BGR images.
    """

    def __init__(self):
        """Initializes an empty particle system."""
        self.particles: list[Particle] = []

    def add_particle(self, particle: Particle) -> None:
        """
        Registers a new particle in the system.

        Args:
            particle: The Particle object to add.
        """
        self.particles.append(particle)

    def update(self, dt: float) -> None:
        """
        Updates the physics state of all active particles and removes expired ones.

        Args:
            dt: Time step delta in seconds.
        """
        if dt <= 0.0:
            return

        for p in self.particles:
            # 1. Update velocities using gravity acceleration
            p.vx += p.gravity[0] * dt
            p.vy += p.gravity[1] * dt

            # 2. Update velocities using drag damping
            # math.pow(drag, dt) provides frame-rate independent drag damping
            damping = math.pow(p.drag, dt)
            p.vx *= damping
            p.vy *= damping

            # 3. Update positions
            p.x += p.vx * dt
            p.y += p.vy * dt

            # 4. Increment lifetime age
            p.age += dt

        # Purge expired particles
        self.particles = [p for p in self.particles if p.age < p.lifetime]

    def draw(self, frame: np.ndarray) -> None:
        """
        Renders all active particles as filled circles onto the frame.
        Applies a simulated alpha fade by scaling brightness to black and shrinking the radius.

        Args:
            frame: BGR frame to draw particles upon.
        """
        h, w, _ = frame.shape

        for p in self.particles:
            # Skip drawing if coordinates are far off-screen
            if not (-50 <= p.x < w + 50 and -50 <= p.y < h + 50):
                continue

            # Calculate lifetime progress ratio (0.0 to 1.0)
            progress = min(1.0, max(0.0, p.age / p.lifetime))
            fade_factor = 1.0 - progress

            # 1. Shrink radius over lifetime
            r = max(1, int(p.radius * fade_factor))

            # 2. Fade BGR color intensity to black (simulating alpha fade on dark backgrounds)
            faded_color = (
                int(p.color[0] * fade_factor),
                int(p.color[1] * fade_factor),
                int(p.color[2] * fade_factor),
            )

            # Draw circle
            cv2.circle(frame, (int(p.x), int(p.y)), r, faded_color, cv2.FILLED)

    def clear(self) -> None:
        """Clears all active particles from the system."""
        self.particles.clear()
