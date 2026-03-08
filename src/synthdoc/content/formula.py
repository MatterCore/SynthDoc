"""LaTeX formula sampling and rendering using matplotlib's mathtext."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


@dataclass
class FormulaResult:
    """Result of formula generation."""

    image: Image.Image
    latex: str


# Formula templates by complexity
SIMPLE_FORMULAS = [
    r"$E = mc^2$",
    r"$a^2 + b^2 = c^2$",
    r"$F = ma$",
    r"$e^{i\pi} + 1 = 0$",
    r"$\sin^2\theta + \cos^2\theta = 1$",
    r"$\frac{d}{dx} x^n = nx^{n-1}$",
    r"$\Delta x = v_0 t + \frac{1}{2}at^2$",
    r"$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$",
    r"$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$",
    r"$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$",
    r"$y = mx + b$",
    r"$V = IR$",
    r"$\log_a(xy) = \log_a x + \log_a y$",
]

MEDIUM_FORMULAS = [
    r"$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$",
    r"$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$",
    r"$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$",
    r"$\mathcal{L}\{f(t)\} = \int_0^\infty f(t)e^{-st}dt$",
    r"$\sum_{n=0}^{\infty} \frac{x^n}{n!} = e^x$",
    r"$\det(A - \lambda I) = 0$",
    r"$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$",
    r"$H = -\sum_{i=1}^{n} p_i \log_2 p_i$",
    r"$\hat{\beta} = (X^T X)^{-1} X^T y$",
    r"$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$",
    r"$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$",
]

COMPLEX_FORMULAS = [
    r"$\int_{-\infty}^{\infty} \frac{\sin(x)}{x} dx = \pi$",
    r"$\frac{\partial \mathcal{L}}{\partial q_i} - \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} = 0$",
    r"$R_{\mu\nu} - \frac{1}{2}g_{\mu\nu}R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}$",
    r"$\psi(x,t) = \sum_n c_n \phi_n(x) e^{-iE_n t/\hbar}$",
    r"$\mathcal{F}\{f(x)\} = \int_{-\infty}^{\infty} f(x) e^{-2\pi i \xi x} dx$",
    r"$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt$",
    r"$\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}$",
    r"$\nabla^2 \phi - \frac{1}{c^2}\frac{\partial^2 \phi}{\partial t^2} = -\frac{\rho}{\epsilon_0}$",
    r"$\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x) dx$",
    r"$KL(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$",
]


class FormulaGenerator:
    """Generates rendered LaTeX formulas using matplotlib's mathtext engine."""

    def __init__(self, complexity: str = "medium") -> None:
        self.complexity = complexity

    def _pick_formula(self, rng: np.random.Generator) -> str:
        """Pick a random formula at the configured complexity level."""
        pools: dict[str, list[str]] = {
            "simple": SIMPLE_FORMULAS,
            "medium": SIMPLE_FORMULAS + MEDIUM_FORMULAS,
            "complex": SIMPLE_FORMULAS + MEDIUM_FORMULAS + COMPLEX_FORMULAS,
        }
        pool = pools.get(self.complexity, MEDIUM_FORMULAS)
        return pool[int(rng.integers(0, len(pool)))]

    def generate(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> FormulaResult:
        """Render a LaTeX formula as an image.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            rng: NumPy random generator.

        Returns:
            FormulaResult with rendered image and LaTeX source.
        """
        latex = self._pick_formula(rng)

        # Calculate figure size in inches (assume 100 dpi for matplotlib rendering)
        render_dpi = 150
        fig_w = width / render_dpi
        fig_h = height / render_dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=render_dpi)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fontsize = max(12, min(36, int(height / 6)))

        ax.text(
            0.5,
            0.5,
            latex,
            fontsize=fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        fig.patch.set_facecolor("white")
        fig.tight_layout(pad=0.5)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        plt.close(fig)

        img = Image.fromarray(img_array).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)

        # Strip the $ signs for the text representation
        latex_clean = latex.strip("$")

        return FormulaResult(image=img, latex=latex_clean)
