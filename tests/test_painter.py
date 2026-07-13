"""
Lightweight Unit Test Script for Painter Class
Verifies stroke creation, continuation, termination, canvas clearing,
and runtime attribute setters.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path to enable importing the shared src modules
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.interaction.painter import Painter


def test_painter_initialization_and_setters():
    print("Testing Painter initialization and setters...")
    painter = Painter(brush_color=(0, 255, 0), brush_size=5)

    # Initial attributes
    assert painter.brush_color == (0, 255, 0)
    assert painter.brush_size == 5
    assert painter.brush_style == "normal"
    assert painter.canvas is None
    assert painter.prev_point is None

    # Test dynamic setters
    painter.set_color((255, 0, 0))
    painter.set_size(12)
    assert painter.brush_color == (255, 0, 0)
    assert painter.brush_size == 12

    # Test style setter
    painter.set_style("neon")
    assert painter.brush_style == "neon"
    painter.set_style("spray")
    assert painter.brush_style == "spray"
    
    # Test invalid style raises ValueError
    try:
        painter.set_style("invalid_style_name")
        assert False, "Expected ValueError for invalid style"
    except ValueError:
        pass

    print("Setters verification: PASSED")


def test_painter_stroke_lifecycle():
    print("\nTesting Painter stroke lifecycle...")
    painter = Painter(brush_color=(0, 255, 0), brush_size=5)

    # Mock frame to initialize canvas shape (480x640)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = painter.render(frame)
    assert painter.canvas is not None
    assert painter.canvas.shape == (480, 640, 3)
    assert np.sum(painter.canvas) == 0  # Canvas starts empty (all black)

    # 1. Stroke creation
    painter.begin_stroke((100, 100))
    assert painter.prev_point == (100, 100)
    print("Stroke creation verification: PASSED")

    # 2. Stroke continuation (Normal style)
    painter.continue_stroke((150, 150))
    assert painter.prev_point == (150, 150)
    assert np.sum(painter.canvas) > 0

    # Stroke continuation in other styles to verify no runtime crashes
    styles = ["marker", "neon", "spray", "rainbow", "eraser"]
    for s in styles:
        painter.set_style(s)
        painter.continue_stroke((180, 180))
        assert painter.prev_point == (180, 180)

    print("Stroke continuation verification: PASSED")

    # 3. Stroke termination
    painter.end_stroke()
    assert painter.prev_point is None
    print("Stroke termination verification: PASSED")

    # 4. Canvas clearing
    painter.clear()
    assert np.sum(painter.canvas) == 0
    print("Canvas clearing verification: PASSED")


if __name__ == "__main__":
    test_painter_initialization_and_setters()
    test_painter_stroke_lifecycle()
    print("\nAll Painter class unit tests passed successfully!")
