"""
Lightweight Unit Test Script for Particle Engine
Verifies particle positioning updates using velocity, gravity,
drag, and automatic particle deletion upon lifetime expiration.
"""

import sys
import math
from pathlib import Path

# Add project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.graphics.particle import Particle
from src.graphics.particle_system import ParticleSystem
from src.graphics.particle_emitter import ParticleEmitter


def test_particle_physics_updates():
    print("Testing Particle System physics update (gravity & drag)...")
    system = ParticleSystem()

    # Create a particle with known initial conditions:
    # x=100.0, y=100.0
    # vx=10.0, vy=20.0
    # gravity=(0.0, 100.0) -> vy acceleration
    # drag=0.9 -> damping of 10% velocity reduction per second
    p = Particle(
        x=100.0,
        y=100.0,
        vx=10.0,
        vy=20.0,
        radius=5.0,
        color=(0, 255, 0),
        lifetime=1.0,
        gravity=(0.0, 100.0),
        drag=0.9,
    )
    system.add_particle(p)

    dt = 0.1

    # Run one update frame
    system.update(dt)

    # 1. Trace vx update:
    # vx_grav = 10.0 + 0.0 * 0.1 = 10.0
    # damping = 0.9^0.1 = 0.9895192582
    # vx_final = 10.0 * 0.9895192582 = 9.895192582
    # x_final = 100.0 + 9.895192582 * 0.1 = 100.9895192582
    expected_damping = math.pow(0.9, dt)
    expected_vx = 10.0 * expected_damping
    expected_x = 100.0 + expected_vx * dt

    # 2. Trace vy update:
    # vy_grav = 20.0 + 100.0 * 0.1 = 30.0
    # vy_final = 30.0 * 0.9895192582 = 29.685577746
    # y_final = 100.0 + 29.685577746 * 0.1 = 102.9685577746
    expected_vy = (20.0 + 100.0 * dt) * expected_damping
    expected_y = 100.0 + expected_vy * dt

    print(f"Computed X: {p.x:.6f} | Expected: {expected_x:.6f}")
    print(f"Computed Y: {p.y:.6f} | Expected: {expected_y:.6f}")
    print(f"Computed Vx: {p.vx:.6f} | Expected: {expected_vx:.6f}")
    print(f"Computed Vy: {p.vy:.6f} | Expected: {expected_vy:.6f}")

    assert math.isclose(p.vx, expected_vx, rel_tol=1e-5)
    assert math.isclose(p.vy, expected_vy, rel_tol=1e-5)
    assert math.isclose(p.x, expected_x, rel_tol=1e-5)
    assert math.isclose(p.y, expected_y, rel_tol=1e-5)
    assert p.age == dt

    print("Physics update verification: PASSED")


def test_particle_expiration():
    print("\nTesting Particle System automatic expiration & deletion...")
    system = ParticleSystem()

    # Create particle with short lifetime
    p = Particle(
        x=100.0,
        y=100.0,
        vx=0.0,
        vy=0.0,
        radius=5.0,
        color=(0, 255, 0),
        lifetime=0.2,  # Lives for 0.2s
    )
    system.add_particle(p)

    # 1. Update by 0.15s (age = 0.15s < 0.2s) -> Particle should still exist
    system.update(0.15)
    assert len(system.particles) == 1
    assert system.particles[0].age == 0.15
    print("Under-lifetime update checks: PASSED")

    # 2. Update by another 0.1s (total age = 0.25s > 0.2s) -> Particle should be deleted
    system.update(0.1)
    assert len(system.particles) == 0
    print("Expired lifetime auto-purging checks: PASSED")


def test_emitter_accumulation():
    print("\nTesting Particle Emitter accumulation logic...")
    system = ParticleSystem()
    emitter = ParticleEmitter(
        emission_rate=50.0,  # 50 particles per second
        speed=100.0,
        spread=360.0,
        color=(0, 255, 0),
        radius=5.0,
        lifetime=1.0,
    )

    # Emit with dt = 0.05 seconds -> expected particles = 50 * 0.05 = 2.5
    # Integer division: spawns 2 particles, keeps 0.5 in accumulator
    emitter.emit((100, 100), system, 0.05)
    assert len(system.particles) == 2
    assert math.isclose(emitter._accumulator, 0.5, rel_tol=1e-5)

    # Emit again with dt = 0.03 seconds -> accumulated = 0.5 + 50 * 0.03 = 0.5 + 1.5 = 2.0
    # Spawns 2 more particles, leaves 0.0 in accumulator
    emitter.emit((100, 100), system, 0.03)
    assert len(system.particles) == 4
    assert math.isclose(emitter._accumulator, 0.0, abs_tol=1e-5)

    print("Emitter accumulation verification: PASSED")


if __name__ == "__main__":
    test_particle_physics_updates()
    test_particle_expiration()
    test_emitter_accumulation()
    print("\nAll Particle Engine unit tests passed successfully!")
