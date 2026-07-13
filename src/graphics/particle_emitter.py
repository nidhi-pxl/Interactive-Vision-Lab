"""
Particle Emitter Module
Defines the ParticleEmitter class responsible for spawning particles dynamically
at interaction coordinates, utilizing a frame-rate independent accumulator.
"""

import math
import random
from src.graphics.particle import Particle
from src.graphics.particle_system import ParticleSystem


class ParticleEmitter:
    """
    Spawns particles into a ParticleSystem with customizable, randomized configurations.
    """

    def __init__(
        self,
        emission_rate: float = 40.0,
        speed: float = 120.0,
        spread: float = 360.0,
        color: tuple[int, int, int] = (0, 255, 255),
        radius: float = 6.0,
        lifetime: float = 1.2,
        gravity: tuple[float, float] = (0.0, 0.0),
        drag: float = 1.0,
    ):
        """
        Initializes the ParticleEmitter configurations.

        Args:
            emission_rate: Number of particles to emit per second.
            speed: Base emission velocity in pixels per second.
            spread: Angle range in degrees. If >= 360, emission is isotropic (all directions).
                    Otherwise, emission is centered pointing upwards (-90 degrees).
            color: BGR tuple representing particle color.
            radius: Base particle radius in pixels.
            lifetime: Base particle lifetime in seconds.
            gravity: Acceleration vector (gx, gy) in pixels/sec^2.
            drag: Drag damping coefficient (fraction of velocity retained per second).
        """
        self.emission_rate = emission_rate
        self.speed = speed
        self.spread = spread
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.gravity = gravity
        self.drag = drag

        # Frame-rate independent accumulator
        self._accumulator = 0.0

    def set_color(self, color: tuple[int, int, int]) -> None:
        """Dynamically updates the particle emission color."""
        self.color = color

    def set_radius(self, radius: float) -> None:
        """Dynamically updates the particle base size radius."""
        self.radius = max(1.0, radius)

    def set_emission_rate(self, rate: float) -> None:
        """Dynamically updates the particle emission rate (particles per second)."""
        self.emission_rate = max(0.0, rate)

    def emit(self, position: tuple[float, float], system: ParticleSystem, dt: float) -> None:
        """
        Emits the correct number of particles based on time delta, adding them to the system.

        Args:
            position: Coordinate (x, y) where particles are generated.
            system: ParticleSystem instance to add particles to.
            dt: Time delta in seconds.
        """
        if dt <= 0.0 or self.emission_rate <= 0.0:
            return

        # Accumulate fractional particles over time to ensure smooth, frame-rate independent output
        self._accumulator += self.emission_rate * dt
        num_to_emit = int(self._accumulator)
        self._accumulator -= num_to_emit

        half_spread = math.radians(self.spread) / 2.0

        for _ in range(num_to_emit):
            # 1. Determine direction based on spread
            if self.spread >= 360.0:
                angle_final = random.uniform(0, 2 * math.pi)
            else:
                # Center directional emission pointing upwards (-pi/2)
                angle_offset = random.uniform(-half_spread, half_spread)
                angle_final = -math.pi / 2.0 + angle_offset

            # 2. Add randomized variations to speed, radius, and lifetime for organic look
            speed_variation = random.uniform(self.speed * 0.7, self.speed * 1.3)
            p_radius = random.uniform(self.radius * 0.7, self.radius * 1.3)
            p_lifetime = random.uniform(self.lifetime * 0.8, self.lifetime * 1.2)

            # Compute velocity vectors (pixels per second)
            vx = speed_variation * math.cos(angle_final)
            vy = speed_variation * math.sin(angle_final)

            # Create and add the particle to the system
            p = Particle(
                x=position[0],
                y=position[1],
                vx=vx,
                vy=vy,
                radius=max(1.0, p_radius),
                color=self.color,
                lifetime=p_lifetime,
                gravity=self.gravity,
                drag=self.drag,
            )
            system.add_particle(p)
