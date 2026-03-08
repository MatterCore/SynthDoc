"""Chart and plot generation using matplotlib."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


@dataclass
class FigureResult:
    """Result of figure generation."""

    image: Image.Image
    chart_type: str
    title: str


CHART_TITLES = [
    "Performance Metrics",
    "Revenue by Quarter",
    "Temperature Distribution",
    "Error Rate Analysis",
    "Population Growth",
    "Sales Comparison",
    "Accuracy vs Epochs",
    "Resource Utilization",
    "Frequency Distribution",
    "Market Share",
    "Convergence Rate",
    "Latency Distribution",
]

X_LABELS = ["Time", "Category", "Epoch", "Month", "Quarter", "Group", "Sample"]
Y_LABELS = ["Value", "Count", "Score", "Rate (%)", "Amount", "Frequency", "Performance"]


class FigureGenerator:
    """Generates chart/plot images using matplotlib."""

    def _make_line_plot(
        self, ax: plt.Axes, rng: np.random.Generator, title: str
    ) -> None:
        """Create a line plot."""
        n_points = int(rng.integers(10, 50))
        n_lines = int(rng.integers(1, 4))
        x = np.arange(n_points)
        for i in range(n_lines):
            y = np.cumsum(rng.standard_normal(n_points)) + i * 5
            ax.plot(x, y, label=f"Series {i + 1}", linewidth=1.5)
        ax.set_xlabel(X_LABELS[int(rng.integers(0, len(X_LABELS)))])
        ax.set_ylabel(Y_LABELS[int(rng.integers(0, len(Y_LABELS)))])
        ax.set_title(title, fontsize=11)
        if n_lines > 1:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _make_bar_chart(
        self, ax: plt.Axes, rng: np.random.Generator, title: str
    ) -> None:
        """Create a bar chart."""
        n_bars = int(rng.integers(4, 10))
        labels = [f"Cat {i + 1}" for i in range(n_bars)]
        values = rng.uniform(10, 100, size=n_bars)
        colors = plt.cm.Set2(np.linspace(0, 1, n_bars))  # type: ignore[attr-defined]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(Y_LABELS[int(rng.integers(0, len(Y_LABELS)))])
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)

    def _make_scatter_plot(
        self, ax: plt.Axes, rng: np.random.Generator, title: str
    ) -> None:
        """Create a scatter plot."""
        n_points = int(rng.integers(30, 150))
        x = rng.standard_normal(n_points)
        y = x * rng.uniform(0.5, 2.0) + rng.standard_normal(n_points) * 0.5
        colors = rng.uniform(0, 1, size=n_points)
        ax.scatter(x, y, c=colors, cmap="viridis", alpha=0.6, s=20)
        ax.set_xlabel(X_LABELS[int(rng.integers(0, len(X_LABELS)))])
        ax.set_ylabel(Y_LABELS[int(rng.integers(0, len(Y_LABELS)))])
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    def _make_pie_chart(
        self, ax: plt.Axes, rng: np.random.Generator, title: str
    ) -> None:
        """Create a pie chart."""
        n_slices = int(rng.integers(3, 8))
        sizes = rng.uniform(5, 40, size=n_slices)
        sizes = sizes / sizes.sum() * 100
        labels = [f"Segment {i + 1}" for i in range(n_slices)]
        colors = plt.cm.Pastel1(np.linspace(0, 1, n_slices))  # type: ignore[attr-defined]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
        ax.set_title(title, fontsize=11)

    def _make_histogram(
        self, ax: plt.Axes, rng: np.random.Generator, title: str
    ) -> None:
        """Create a histogram."""
        n_points = int(rng.integers(100, 500))
        data = rng.standard_normal(n_points) * rng.uniform(1, 5) + rng.uniform(-10, 10)
        ax.hist(data, bins=int(rng.integers(10, 30)), color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel(X_LABELS[int(rng.integers(0, len(X_LABELS)))])
        ax.set_ylabel("Frequency")
        ax.set_title(title, fontsize=11)

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> FigureResult:
        """Generate a chart/plot as an image.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            rng: NumPy random generator.

        Returns:
            FigureResult with rendered chart image, type, and title.
        """
        render_dpi = 150
        fig_w = width / render_dpi
        fig_h = height / render_dpi

        chart_makers = [
            ("line", self._make_line_plot),
            ("bar", self._make_bar_chart),
            ("scatter", self._make_scatter_plot),
            ("pie", self._make_pie_chart),
            ("histogram", self._make_histogram),
        ]
        idx = int(rng.integers(0, len(chart_makers)))
        chart_type, maker = chart_makers[idx]
        title = CHART_TITLES[int(rng.integers(0, len(CHART_TITLES)))]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=render_dpi)
        maker(ax, rng, title)
        fig.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        plt.close(fig)

        img = Image.fromarray(img_array).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)

        return FigureResult(image=img, chart_type=chart_type, title=title)
