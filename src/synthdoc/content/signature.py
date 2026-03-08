"""Signature generation using random bezier curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class SignatureResult:
    """Result of signature generation."""

    image: Image.Image


def _bezier_curve(
    points: np.ndarray, num_steps: int = 100
) -> np.ndarray:
    """Evaluate a cubic bezier curve defined by control points.

    Args:
        points: Control points array of shape (4, 2).
        num_steps: Number of interpolation steps.

    Returns:
        Array of shape (num_steps, 2) with curve coordinates.
    """
    t = np.linspace(0, 1, num_steps).reshape(-1, 1)
    # Cubic bezier: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    coeffs = np.column_stack([
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3,
    ])
    return coeffs @ points


class SignatureGenerator:
    """Generates realistic-looking signature images from random bezier curves."""

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> SignatureResult:
        """Generate a signature image.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            rng: NumPy random generator.

        Returns:
            SignatureResult with rendered signature image.
        """
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Signature parameters
        num_strokes = int(rng.integers(3, 8))
        padding_x = width * 0.1
        padding_y = height * 0.15

        # Ink color (dark blue-black)
        ink_r = int(rng.integers(0, 30))
        ink_g = int(rng.integers(0, 30))
        ink_b = int(rng.integers(40, 90))
        ink_color = (ink_r, ink_g, ink_b)

        # Generate connected bezier strokes
        x_cursor = padding_x + rng.uniform(0, width * 0.1)
        y_center = height / 2

        for stroke_idx in range(num_strokes):
            # Generate 4 control points for cubic bezier
            x_span = (width - 2 * padding_x) / num_strokes
            x_start = x_cursor
            x_end = x_cursor + x_span * rng.uniform(0.6, 1.2)

            y_amplitude = (height - 2 * padding_y) / 2 * rng.uniform(0.3, 1.0)

            control_points = np.array([
                [x_start, y_center + rng.normal(0, y_amplitude * 0.3)],
                [
                    x_start + (x_end - x_start) * 0.33,
                    y_center + rng.uniform(-y_amplitude, y_amplitude),
                ],
                [
                    x_start + (x_end - x_start) * 0.66,
                    y_center + rng.uniform(-y_amplitude, y_amplitude),
                ],
                [x_end, y_center + rng.normal(0, y_amplitude * 0.3)],
            ])

            # Evaluate bezier curve
            num_steps = int(rng.integers(60, 120))
            curve = _bezier_curve(control_points, num_steps)

            # Draw the curve with varying stroke width
            base_width = max(1, int(rng.integers(2, 5)))
            for i in range(len(curve) - 1):
                # Vary width along the stroke (pressure simulation)
                t = i / max(1, len(curve) - 1)
                width_mult = 1.0 - 0.5 * abs(2 * t - 1)  # Thicker in middle
                stroke_w = max(1, int(base_width * width_mult + rng.normal(0, 0.3)))

                x1, y1 = curve[i]
                x2, y2 = curve[i + 1]

                # Clamp to image bounds
                x1 = np.clip(x1, 0, width - 1)
                y1 = np.clip(y1, 0, height - 1)
                x2 = np.clip(x2, 0, width - 1)
                y2 = np.clip(y2, 0, height - 1)

                draw.line(
                    [(float(x1), float(y1)), (float(x2), float(y2))],
                    fill=ink_color,
                    width=stroke_w,
                )

            x_cursor = float(x_end) + rng.uniform(-5, 10)

        # Optionally add a dot or flourish at the end
        if rng.random() > 0.5:
            dot_x = float(np.clip(x_cursor + rng.uniform(5, 20), 0, width - 10))
            dot_y = float(y_center + rng.normal(0, height * 0.1))
            dot_y = float(np.clip(dot_y, 5, height - 5))
            dot_r = int(rng.integers(2, 5))
            draw.ellipse(
                [dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r],
                fill=ink_color,
            )

        # Optionally add an underline
        if rng.random() > 0.6:
            line_y = float(y_center + height * 0.25 + rng.normal(0, 5))
            line_y = float(np.clip(line_y, 0, height - 3))
            draw.line(
                [(padding_x, line_y), (width - padding_x, line_y)],
                fill=ink_color,
                width=max(1, int(rng.integers(1, 3))),
            )

        return SignatureResult(image=img)
