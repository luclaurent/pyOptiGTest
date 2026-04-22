"""Generate pictures for all test functions (surface + contour + gradients).

This test module mirrors the MATLAB ``various/pictures.m`` script.
Each 2D-compatible function gets a figure with:
  - 3D surface plot of the objective
  - Contour plot of the objective
  - 3D surface plots of the X and Y gradients
  - Contour plots of the X and Y gradients

The output is written to ``wiki/Figures/`` (relative to the project root)
and companion Markdown files are generated for GitHub wiki pages.

Run with::

    pytest tests/test_generate_pictures.py -v --gen-pictures

The ``--gen-pictures`` flag is required to actually generate images (the tests
are skipped by default so that normal CI runs are not slowed down).
"""

import pathlib
import re
import textwrap
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from conftest import ALL_FUNCTION_NAMES, REQUIRED_DIM, load_function

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
WIKI_DIR: pathlib.Path = ROOT / "wiki"
FIG_DIR: pathlib.Path = WIKI_DIR / "Figures"
GRID_SIZE: int = 200

# ---------------------------------------------------------------------------
# Design-space bounds lookup.
# Keys are function names; values are either:
#   (lo, hi)           – same bounds for all variables, or
#   [(lo1,hi1), ...]   – per-variable bounds.
# Functions not listed fall back to docstring parsing, then to (-5, 5).
# ---------------------------------------------------------------------------

BOUNDS: dict[str, tuple | list] = {
    # Ackley family
    "funAckley2": (-32, 32),
    "funAckley3": (-32, 32),
    "funAckley4": (-35, 35),
    # Adjiman
    "funAdjiman": (-1, 2),
    # AHE
    "funAHE": (-10, 10),
    # Alpine
    "funAlpine1": (-10, 10),
    "funAlpine2": (0, 10),
    # AMGM
    "funAMGM": (0, 10),
    # Bartels-Conn
    "funBartelsConn": (-500, 500),
    # Beale
    "funBeale": (-4.5, 4.5),
    # Biggs
    "funBiggsExp2": (0, 20),
    # Bird
    "funBird": (-2 * np.pi, 2 * np.pi),
    # Bohachevsky
    "funBohachevsky1": (-100, 100),
    "funBohachevsky2": (-100, 100),
    "funBohachevsky3": (-100, 100),
    # Booth
    "funBooth": (-10, 10),
    # Box-Betts
    "funBoxBetts": [(1, 10), (10, 20), (1, 5)],
    # Branin
    "funBranin1": [(-5, 10), (0, 15)],
    "funBranin2": [(-5, 10), (0, 15)],
    # Brent
    "funBrent": (-10, 10),
    # Brown
    "funBrown": (-1, 4),
    # Bukin family
    "funBukin01": [(-15, -5), (-3, 3)],
    "funBukin02": [(-15, -5), (-3, 3)],
    "funBukin03": [(-15, -5), (-3, 3)],
    "funBukin04": [(-15, -5), (-3, 3)],
    "funBukin05": [(-15, -5), (-3, 3)],
    "funBukin06": [(-15, -5), (-3, 3)],
    "funBukin07": [(-15, -5), (-3, 3)],
    "funBukin08": [(-15, -5), (-3, 3)],
    "funBukin09": [(-15, -5), (-3, 3)],
    "funBukin10": [(-15, -5), (-3, 3)],
    "funBukin11": [(-15, -5), (-3, 3)],
    "funBukin12": [(-15, -5), (-3, 3)],
    "funBukin13": [(-15, -5), (-3, 3)],
    "funBukin14": [(-15, -5), (-3, 3)],
    "funBukin15": [(-15, -5), (-3, 3)],
    "funBukin16": [(-15, -5), (-3, 3)],
    "funBukin17": [(-15, -5), (-3, 3)],
    "funBukin18": [(-15, -5), (-3, 3)],
    "funBukin19": [(-15, -5), (-3, 3)],
    "funBukin20": [(-15, -5), (-3, 3)],
    # Camelback
    "funCamelbackSixHump": [(-3, 3), (-2, 2)],
    "funCamelbackThreeHump": (-5, 5),
    # Carrom-Table
    "funCarromTable": (-10, 10),
    # Chen
    "funChenBird": (-500, 500),
    "funChenV": (-500, 500),
    # Chichinadze
    "funChichinadze": (-30, 30),
    # Chung-Reynolds
    "funChungReynolds": (-100, 100),
    # Cigar
    "funCigar": (-100, 100),
    # Colville (4D – skip for 2D pictures)
    # Corana
    "funCorana": (-5, 5),
    # Cosine Mixture
    "funCosineMixture": (-1, 1),
    # Cross-in-Tray
    "funCrossInTray": (-10, 10),
    # Cross-Leg Table
    "funCrossLegTable": (-10, 10),
    # Crowned Cross
    "funCrownedCross": (-10, 10),
    # Csendes
    "funCsendes": (-1, 1),
    # Cst
    "funCst": (-5, 5),
    # Cube
    "funCube": (-10, 10),
    # Custom
    "funCustom01": (-10, 10),
    "funCustom02": (-10, 10),
    "funCustom03": (-10, 10),
    "funCustom04": (-10, 10),
    "funCustom05": (-10, 10),
    "funCustom06": (-10, 10),
    "funCustom07": (-10, 10),
    # Damavandi
    "funDamavandi": (0, 14),
    # Deb
    "funDeb1": (0, 1),
    "funDeb2": (0, 1),
    "funDeb3": (0, 1),
    "funDeb4": (0, 1),
    # Decanomial
    "funDecanomial": (-10, 10),
    # Deceptive
    "funDeceptive": (0, 1),
    # Decckers-Aarts
    "funDeckkersAarts": (-20, 20),
    # Deflected Corrugated Spring
    "funDeflectedCorrugatedSpring": (0, 10),
    # De Jong (Sphere)
    "funDejong": (-5.12, 5.12),
    # Dixon
    "funDixon": (-10, 10),
    # Dixon-Price
    "funDixonPrice": (-10, 10),
    # Drop-Wave
    "funDropWave": (-5.12, 5.12),
    # Easom
    "funEasom": (-100, 100),
    # Egg-Crate
    "funEggCrate": (-5, 5),
    # Egg Holder
    "funEggHolder": (-512, 512),
    # El-Attar
    "funElAttarVidyasogarDutta": (-100, 100),
    # EX1
    "funEX1": (-5, 5),
    # Exponential
    "funExponential": (-1, 1),
    # Freudenstein-Roth
    "funFreudensteinRoth": (-10, 10),
    # Goldstein-Price
    "funGoldsteinPrice": (-2, 2),
    # Griewank
    "funGriewank": (-100, 100),
    # Hansen
    "funHansen": (-10, 10),
    # Himmelblau
    "funHimmelblau": (-5, 5),
    # Hosaki
    "funHosaki": (0, 5),
    # Jennrich-Sampson
    "funJennrichSampson": (-1, 1),
    # Judge
    "funJudge": (-10, 10),
    # Keane
    "funKeane": (0, 10),
    # Langermann
    "funLangermann52": (0, 10),
    # Leon
    "funLeon": (-1.2, 1.2),
    # Levy
    "funLevy05": (-10, 10),
    "funLevy13": (-10, 10),
    # Matyas
    "funMatyas": (-10, 10),
    # McCormick
    "funMcCormick": [(-1.5, 4), (-3, 4)],
    # Michalewicz
    "funMichalewicz": (0, np.pi),
    # Mishra
    "funMishra03": (-10, 10),
    "funMishra04": (-10, 10),
    "funMishra05": (-10, 10),
    "funMishra06": (-10, 10),
    "funMishra08": (-10, 10),
    "funMishra10": (-10, 10),
    # Mystery
    "funMystery": (-5, 5),
    # New functions
    "funNewFunction1": (-10, 10),
    "funNewFunction2": (-10, 10),
    "funNewFunction3": (-10, 10),
    # Null
    "funNull": (-5, 5),
    # Parsopoulos
    "funParsopoulos": (-5, 5),
    # Peaks
    "funPeaks": (-3, 3),
    "funPeaksN": (-3, 3),
    # Pen Holder
    "funPenHolder": (-11, 11),
    # Periodic
    "funPeriodic": (-10, 10),
    # Price
    "funPrice2": (-10, 10),
    "funPrice3": (-500, 500),
    "funPrice4": (-500, 500),
    # Quadratic
    "funQuadratic": (-10, 10),
    # Quartic
    "funQuartic": (-1.28, 1.28),
    # Rastrigin
    "funRastrigin": (-5.12, 5.12),
    # Rosenbrock
    "funRosenbrock": (-5, 10),
    "funRosenbrockM": (-5, 10),
    # Rotated Ellipse
    "funRotatedEllipse1": (-500, 500),
    "funRotatedEllipse2": (-500, 500),
    # Rump
    "funRump": (-500, 500),
    # Schaffer
    "funSchaffer1": (-100, 100),
    "funSchaffer2": (-100, 100),
    "funSchaffer3": (-100, 100),
    "funSchaffer4": (-100, 100),
    "funSchaffer6": (-100, 100),
    # Schwefel
    "funSchwefel": (-500, 500),
    "funSchwefel06": (-100, 100),
    "funSchwefel36": (0, 500),
    # Shubert
    "funShubert1": (-10, 10),
    # Sine Envelope
    "funSineEnveloppe": (-100, 100),
    # Sphere
    "funSphere": (-5.12, 5.12),
    # Stochastic
    "funStochastic": (-5, 5),
    # Stretched V
    "funStretchedV": (-10, 10),
    # Styblinski-Tang
    "funStyblinskiTang": (-5, 5),
    # Sum Square
    "funSumSquare": (-10, 10),
    # Treccani
    "funTreccani": (-5, 5),
    # Trefethen
    "funTrefethen": (-10, 10),
    # Trid
    "funTrid": (-4, 4),
    # Trigonometric
    "funTrigonometric3": (-5, 5),
    # Tripod
    "funTripod": (-100, 100),
    # Tube-Holder
    "funTubeHolder": (-10, 10),
    # Ursem
    "funUrsem01": [(-2.5, 3), (-2, 2)],
    "funUrsem4": (-2, 2),
    "funUrsemWaves": [(-0.9, 1.2), (-1.2, 1.2)],
    # Wayburn-Seader
    "funWayburnSeader1": (-5, 5),
    "funWayburnSeader2": (-500, 500),
    # Whitley
    "funWhitley": (-10.24, 10.24),
    # Xin-She-Yang
    "funXinSheYang1": (-5, 5),
    "funXinSheYang2": (-2 * np.pi, 2 * np.pi),
    "funXinSheYang3": (-20, 20),
    "funXinSheYang4": (-10, 10),
    # Zettl
    "funZettl": (-5, 10),
    # Zimmerman
    "funZimmerman": (0, 100),
    # Zirilli
    "funZirilli": (-10, 10),
}

# Default fallback bounds when nothing else is available
DEFAULT_BOUNDS: tuple[float, float] = (-5, 5)

# Functions known to misbehave (NaN/Inf on regular grids) – skip for pictures
SKIP_PICTURE: set[str] = {
    "funGear",          # integer-domain
    "funGulfResearch",  # 3-D, NaN-prone
    "funPaviani",       # NaN-prone
    "funCola",          # 17-D only
    "funXor",           # 9-D only
    "funWatson",        # 6-D only
    "funDolan",         # 5-D only
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bounds_2d(name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((xlo, xhi), (ylo, yhi)) for a 2-variable evaluation."""
    if name in BOUNDS:
        b = BOUNDS[name]
        if isinstance(b, list):
            # per-variable bounds – take first two
            return tuple(b[0]), tuple(b[1])
        else:
            return b, b
    # Try parsing docstring
    bounds = _parse_docstring_bounds(name)
    if bounds is not None:
        return bounds
    return DEFAULT_BOUNDS, DEFAULT_BOUNDS


def _parse_docstring_bounds(name: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Attempt to extract symmetric bounds from the function docstring."""
    try:
        fn = load_function(name)
        doc = fn.__doc__ or ""
    except Exception:
        return None

    # Pattern: "design space" followed by -N<xi<N  or  -N < xi < N
    m = re.search(
        r"[Dd]esign\s+space[:\s]*(-?[\d.]+(?:e[+-]?\d+)?)\s*<\s*x\w*\s*<\s*(-?[\d.]+(?:e[+-]?\d+)?)",
        doc,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi), (lo, hi)
    # pi-based bounds
    m = re.search(
        r"[Dd]esign\s+space[:\s]*(-?[\d.]*)\s*\*?\s*pi\s*<\s*x\w*\s*<\s*(-?[\d.]*)\s*\*?\s*pi",
        doc,
    )
    if m:
        lo_s = m.group(1) or "1"
        hi_s = m.group(2) or "1"
        # Handle bare "-" meaning "-1"
        lo = float("-1" if lo_s == "-" else lo_s) * np.pi
        hi = float("-1" if hi_s == "-" else hi_s) * np.pi
        return (lo, hi), (lo, hi)
    return None


def _is_2d_compatible(name: str) -> bool:
    """Return True if the function can be evaluated in 2 dimensions."""
    if name in SKIP_PICTURE:
        return False
    req = REQUIRED_DIM.get(name, 1)
    # Functions with required dim <= 2, or any-dim functions (default dim=5 means "any")
    return req <= 2


def _get_2d_function_names() -> list[str]:
    """Return sorted list of function names that support 2D evaluation."""
    return sorted(n for n in ALL_FUNCTION_NAMES if _is_2d_compatible(n))


FUNCTION_NAMES_2D: list[str] = _get_2d_function_names()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def output_dirs() -> pathlib.Path:
    """Create output directories and return the base figure dir."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


# ---------------------------------------------------------------------------
# Tests – picture generation
# ---------------------------------------------------------------------------

@pytest.mark.genpictures
class TestGeneratePictures:
    """Generate 2D surface / contour / gradient pictures for every compatible function."""

    @pytest.mark.parametrize("name", FUNCTION_NAMES_2D)
    def test_generate_picture(self, name: str, output_dirs: pathlib.Path) -> None:
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fn = load_function(name)
        (xlo, xhi), (ylo, yhi) = _get_bounds_2d(name)

        # Build evaluation grid
        xv = np.linspace(xlo, xhi, GRID_SIZE)
        yv = np.linspace(ylo, yhi, GRID_SIZE)
        Xm, Ym = np.meshgrid(xv, yv)
        XX = np.column_stack([Xm.ravel(), Ym.ravel()])

        # Evaluate function + gradients
        try:
            result = fn(XX, grad=True)
            if isinstance(result, tuple) and len(result) == 2:
                Z, dZ = result
                has_grad = True
            else:
                Z = result if not isinstance(result, tuple) else result[0]
                has_grad = False
        except Exception:
            Z = fn(XX)
            has_grad = False
            dZ = None

        Z = np.asarray(Z).ravel()
        Zm = Z.reshape(Xm.shape)

        # Clamp extreme values for better visualisation
        finite = Zm[np.isfinite(Zm)]
        if finite.size > 0:
            q01, q99 = np.percentile(finite, [1, 99])
            Zm_plot = np.clip(Zm, q01 - 0.1 * abs(q01), q99 + 0.1 * abs(q99))
        else:
            Zm_plot = Zm

        if has_grad:
            dZ = np.asarray(dZ)
            dZx = dZ[:, 0].reshape(Xm.shape)
            dZy = dZ[:, 1].reshape(Xm.shape)
        else:
            dZx = dZy = None

        # Number of subplot rows: 2 (surface+contour) per quantity
        n_cols = 3 if has_grad else 1
        fig = plt.figure(figsize=(7 * n_cols, 12), constrained_layout=True)
        fig.suptitle(name, fontsize=16, fontweight="bold")

        # --- Objective surface ---
        ax1 = fig.add_subplot(2, n_cols, 1, projection="3d")
        ax1.plot_surface(Xm, Ym, Zm_plot, cmap=cm.viridis, linewidth=0,
                         antialiased=False, alpha=0.9, rcount=100, ccount=100)
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.set_title("Objective")

        # --- Objective contour ---
        ax4 = fig.add_subplot(2, n_cols, n_cols + 1)
        n_levels = 50
        cs = ax4.contourf(Xm, Ym, Zm_plot, levels=n_levels, cmap=cm.viridis)
        fig.colorbar(cs, ax=ax4, shrink=0.8)
        ax4.set_xlabel("$x_1$")
        ax4.set_ylabel("$x_2$")
        ax4.set_title("Objective (contour)")
        ax4.set_aspect("equal", adjustable="box")

        if has_grad:
            # Clamp gradients
            for arr, label, col_offset in [
                (dZx, "Grad. $x_1$", 1),
                (dZy, "Grad. $x_2$", 2),
            ]:
                finite_g = arr[np.isfinite(arr)]
                if finite_g.size > 0:
                    gq01, gq99 = np.percentile(finite_g, [1, 99])
                    arr_plot = np.clip(arr, gq01 - 0.1 * abs(gq01), gq99 + 0.1 * abs(gq99))
                else:
                    arr_plot = arr

                # Surface
                ax_s = fig.add_subplot(2, n_cols, 1 + col_offset, projection="3d")
                ax_s.plot_surface(Xm, Ym, arr_plot, cmap=cm.plasma, linewidth=0,
                                  antialiased=False, alpha=0.9, rcount=100, ccount=100)
                ax_s.set_xlabel("$x_1$")
                ax_s.set_ylabel("$x_2$")
                ax_s.set_title(label)

                # Contour
                ax_c = fig.add_subplot(2, n_cols, n_cols + 1 + col_offset)
                cs_g = ax_c.contourf(Xm, Ym, arr_plot, levels=n_levels, cmap=cm.plasma)
                fig.colorbar(cs_g, ax=ax_c, shrink=0.8)
                ax_c.set_xlabel("$x_1$")
                ax_c.set_ylabel("$x_2$")
                ax_c.set_title(f"{label} (contour)")
                ax_c.set_aspect("equal", adjustable="box")

        # Save
        out_path = output_dirs / f"{name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        assert out_path.exists(), f"Failed to create {out_path}"


# ---------------------------------------------------------------------------
# Tests – wiki markdown generation
# ---------------------------------------------------------------------------

@pytest.mark.genpictures
class TestGenerateWikiMarkdown:
    """Generate Markdown pages listing all function pictures for the GitHub wiki."""

    def test_generate_markdown(self, output_dirs: pathlib.Path) -> None:
        matplotlib = pytest.importorskip("matplotlib")

        lines = [
            "# Optimization Test Functions Gallery\n",
            "",
            "This page shows all 2D-compatible test functions available in "
            "**pyOptiGTest**.\n",
            "",
            "Each figure displays the objective surface (3D and contour) "
            "together with analytical gradients when available.\n",
            "",
            "---\n",
            "",
        ]

        for name in FUNCTION_NAMES_2D:
            png = FIG_DIR / f"{name}.png"
            if png.exists():
                rel = f"Figures/{name}.png"
                lines.append(f"## {name}\n")
                lines.append("")
                lines.append(f"![{name}]({rel})\n")
                lines.append("")
                lines.append("---\n")
                lines.append("")

        md_path = WIKI_DIR / "Functions-Gallery.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        assert md_path.exists()


# ---------------------------------------------------------------------------
# Tests – individual category markdown files (like MATLAB script)
# ---------------------------------------------------------------------------

@pytest.mark.genpictures
class TestGenerateCategoryMarkdown:
    """Generate per-category Markdown files (Unconstrained, etc.)."""

    def test_generate_unconstrained_md(self, output_dirs: pathlib.Path) -> None:
        matplotlib = pytest.importorskip("matplotlib")

        lines = [
            "# Unconstrained Test Functions\n",
            "",
        ]
        for name in FUNCTION_NAMES_2D:
            png = FIG_DIR / f"{name}.png"
            if png.exists():
                rel = f"Figures/{name}.png"
                lines.append(f"## {name}\n")
                lines.append("")
                lines.append(f"![{name}]({rel})\n")
                lines.append("")

        md_path = WIKI_DIR / "Unconstrained.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        assert md_path.exists()
