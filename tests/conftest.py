"""Shared fixtures and helpers for pyOptiGTest tests."""

import importlib
import importlib.util
import os
import pathlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CLI option for picture generation
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--gen-pictures",
        action="store_true",
        default=False,
        help="Actually generate wiki pictures (slow).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gen-pictures"):
        return
    skip = pytest.mark.skip(reason="need --gen-pictures option to run")
    for item in items:
        if "genpictures" in item.keywords:
            item.add_marker(skip)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
FUNCTIONS_DIR = ROOT / "pyOptiGTest" / "functions"

# ---------------------------------------------------------------------------
# Dimension map – minimum number of variables required by each function.
# Functions not listed here accept any dimension >= 1.
# ---------------------------------------------------------------------------

REQUIRED_DIM: dict[str, int] = {
    "funAckley3": 2,
    "funAdjiman": 2,
    "funBartelsConn": 2,
    "funBeale": 2,
    "funBiggsExp2": 2,
    "funBiggsExp3": 3,
    "funBiggsExp4": 4,
    "funBiggsExp5": 5,
    "funBiggsExp6": 6,
    "funBird": 2,
    "funBohachevsky1": 2,
    "funBohachevsky2": 2,
    "funBohachevsky3": 2,
    "funBooth": 2,
    "funBoxBetts": 3,
    "funBrad": 3,
    "funBranin1": 2,
    "funBranin2": 2,
    "funBrent": 2,
    "funBrown": 2,
    "funBukin01": 2,
    "funBukin02": 2,
    "funBukin03": 2,
    "funBukin04": 2,
    "funBukin05": 2,
    "funBukin06": 2,
    "funBukin07": 2,
    "funBukin08": 2,
    "funBukin09": 2,
    "funBukin10": 2,
    "funBukin11": 2,
    "funBukin12": 2,
    "funBukin13": 2,
    "funBukin14": 2,
    "funBukin15": 2,
    "funBukin16": 2,
    "funBukin17": 2,
    "funBukin18": 2,
    "funBukin19": 2,
    "funBukin20": 2,
    "funCamelbackSixHump": 2,
    "funCamelbackThreeHump": 2,
    "funCarromTable": 2,
    "funChenBird": 2,
    "funChenV": 2,
    "funChichinadze": 2,
    "funCola": 17,
    "funColville": 4,
    "funCrossInTray": 2,
    "funCrossLegTable": 2,
    "funCrownedCross": 2,
    "funCube": 2,
    "funDamavandi": 2,
    "funDeVilliersGlasser1": 4,
    "funDeVilliersGlasser2": 5,
    "funDecanomial": 2,
    "funDeckkersAarts": 2,
    "funDixon": 2,
    "funDolan": 5,
    "funEX1": 2,
    "funEasom": 2,
    "funEggCrate": 2,
    "funElAttarVidyasogarDutta": 2,
    "funExp2": 2,
    "funExp3": 3,
    "funExp4": 4,
    "funExp5": 5,
    "funExp6": 6,
    "funFreudensteinRoth": 2,
    "funGear": 4,
    "funGoldsteinPrice": 2,
    "funGulfResearch": 3,
    "funHansen": 2,
    "funHartmann3": 3,
    "funHartmann6": 6,
    "funHelicalValley": 3,
    "funHimmelblau": 2,
    "funHolzman": 3,
    "funHosaki": 2,
    "funJennrichSampson": 2,
    "funJudge": 2,
    "funKeane": 2,
    "funKowalik": 4,
    "funLangermann52": 2,
    "funLeon": 2,
    "funLevy05": 2,
    "funLevy13": 2,
    "funMatyas": 2,
    "funMcCormick": 2,
    "funMieleCantrell": 4,
    "funMishra03": 2,
    "funMishra04": 2,
    "funMishra05": 2,
    "funMishra06": 2,
    "funMishra08": 2,
    "funMishra09": 3,
    "funMishra10": 2,
    "funMystery": 2,
    "funNewFunction1": 2,
    "funNewFunction2": 2,
    "funNewFunction3": 2,
    "funParsopoulos": 2,
    "funPeaks": 2,
    "funPeaksN": 2,
    "funPenHolder": 2,
    "funPeriodic": 2,
    "funPowell": 4,
    "funPrice2": 2,
    "funPrice3": 2,
    "funPrice4": 2,
    "funQuadratic": 2,
    "funRosenbrockM": 2,
    "funRotatedEllipse1": 2,
    "funRotatedEllipse2": 2,
    "funRump": 2,
    "funSchaffer1": 2,
    "funSchaffer2": 2,
    "funSchaffer3": 2,
    "funSchaffer4": 2,
    "funSchaffer6": 2,
    "funSchmidtVetters": 3,
    "funSchwefel06": 2,
    "funSchwefel36": 2,
    "funShekel05": 4,
    "funShekel07": 4,
    "funShekel10": 4,
    "funShubert1": 2,
    "funSineEnveloppe": 2,
    "funStretchedV": 2,
    "funTreccani": 2,
    "funTrefethen": 2,
    "funTrid": 2,
    "funTrigonometric3": 2,
    "funTripod": 2,
    "funTubeHolder": 2,
    "funUrsem01": 2,
    "funUrsem4": 2,
    "funUrsemWaves": 2,
    "funWatson": 6,
    "funWayburnSeader1": 2,
    "funWayburnSeader2": 2,
    "funWeibull": 3,
    "funWolfe": 3,
    "funXor": 9,
    "funZettl": 2,
    "funZimmerman": 2,
    "funZirilli": 2,
    # Constrained functions
    "funCons1": 2,
    "funCons2": 2,
    "funConsSimionescu": 2,
    "funConsTownsend": 2,
    "funDisk2": 2,
    "funDisk25": 2,
    "funSimionescu": 2,
    "funTownsend": 2,
    # Multi-objective: BinhKorn
    "funObjKornBinh1": 2,
    "funObjKornBinh2": 2,
    "funConsKornBinh1": 2,
    "funConsKornBinh2": 2,
    # Multi-objective: ChakongHaimes
    "funObjChakongHaimes1": 2,
    "funObjChakongHaimes2": 2,
    "funConsChakongHaimes1": 2,
    "funConsChakongHaimes2": 2,
    # Multi-objective: FonsecaFleming
    "funObjFonsecaFleming1": 3,
    "funObjFonsecaFleming2": 3,
    # Multi-objective: TestFun4
    "funObjTestFun41": 2,
    "funObjTestFun42": 2,
    "funConsTestFun41": 2,
    "funConsTestFun42": 2,
    "funConsTestFun43": 2,
    # Multi-objective: Kursawe
    "funObjKursawe1": 3,
    "funObjKursawe2": 3,
    # Multi-objective: MultiSchaffer
    "funObjMultiSchaffer11": 1,
    "funObjMultiSchaffer12": 1,
    "funObjMultiSchaffer21": 1,
    "funObjMultiSchaffer22": 1,
    # Multi-objective: Poloni
    "funObjPoloni1": 2,
    "funObjPoloni2": 2,
    # Multi-objective: ZDT
    "funObjZDT11": 30,
    "funObjZDT12": 30,
    "funObjZDT21": 30,
    "funObjZDT22": 30,
    "funObjZDT31": 30,
    "funObjZDT32": 30,
    "funObjZDT41": 10,
    "funObjZDT42": 10,
    "funObjZDT61": 10,
    "funObjZDT62": 10,
    # Multi-objective: OsyczkaKundu
    "funObjOsyczkaKundu1": 6,
    "funObjOsyczkaKundu2": 6,
    "funConsOsyczkaKundu1": 6,
    "funConsOsyczkaKundu2": 6,
    "funConsOsyczkaKundu3": 6,
    "funConsOsyczkaKundu4": 6,
    "funConsOsyczkaKundu5": 6,
    "funConsOsyczkaKundu6": 6,
    # Multi-objective: CTP1
    "funObjCTP11": 2,
    "funObjCTP12": 2,
    "funConsCTP11": 2,
    "funConsCTP12": 2,
    # Multi-objective: ConstrEx
    "funObjConstrEx1": 2,
    "funObjConstrEx2": 2,
    "funConsConstrEx1": 2,
    "funConsConstrEx2": 2,
    # Multi-objective: Viennet
    "funObjViennet1": 2,
    "funObjViennet2": 2,
    "funObjViennet3": 2,
}

# Default dimension when no specific requirement
DEFAULT_DIM = 5

# ---------------------------------------------------------------------------
# Known optima: (function_name, x_optimal, f_optimal, tolerance)
# ---------------------------------------------------------------------------

KNOWN_OPTIMA: list[tuple[str, np.ndarray, float, float]] = [
    ("funDejong", np.zeros((1, 5)), 0.0, 1e-10),
    ("funRosenbrock", np.ones((1, 5)), 0.0, 1e-10),
    ("funHartmann3", np.array([[0.114614, 0.555649, 0.852547]]), -3.86278, 1e-3),
    ("funHartmann6", np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]), -3.32237, 1e-3),
    ("funShekel05", np.array([[4.0, 4.0, 4.0, 4.0]]), -10.1532, 1e-2),
    ("funGriewank", np.zeros((1, 5)), 0.0, 1e-10),
    ("funTrid", np.array([[6.0, 10.0, 12.0, 12.0, 10.0, 6.0]]), -50.0, 1e-6),
    ("funMichalewicz", np.array([[2.20, 1.57]]), -1.8013, 1e-2),
    ("funPowell", np.zeros((1, 4)), 0.0, 1e-10),
    ("funWhitley", np.ones((1, 3)), 0.0, 1e-6),
    ("funAlpine1", np.zeros((1, 5)), 0.0, 1e-10),
    ("funBooth", np.array([[1.0, 3.0]]), 0.0, 1e-10),
    ("funMatyas", np.zeros((1, 2)), 0.0, 1e-10),
    ("funSphere", np.zeros((1, 5)), 0.0, 1e-10),
    ("funBeale", np.array([[3.0, 0.5]]), 0.0, 1e-10),
    ("funGoldsteinPrice", np.array([[0.0, -1.0]]), 3.0, 1e-10),
    ("funHimmelblau", np.array([[3.0, 2.0]]), 0.0, 1e-6),
    ("funLeon", np.array([[1.0, 1.0]]), 0.0, 1e-10),
    ("funCst", np.zeros((1, 3)), 10.0, 1e-10),
    ("funNull", np.zeros((1, 3)), 0.0, 1e-10),
    ("funEasom", np.array([[np.pi, np.pi]]), -1.0, 1e-6),
    ("funSumSquare", np.zeros((1, 5)), 0.0, 1e-10),
    ("funChungReynolds", np.zeros((1, 5)), 0.0, 1e-10),
    ("funExponential", np.zeros((1, 5)), -1.0, 1e-6),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_function_names() -> list[str]:
    """Return sorted list of all function names from the functions directory."""
    names = []
    for f in sorted(FUNCTIONS_DIR.glob("fun*.py")):
        name = f.stem
        if name != "__init__":
            names.append(name)
    return names


def load_function(name: str):
    """Import a single function by name from its file."""
    path = FUNCTIONS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def dim_for(name: str) -> int:
    """Return the number of dimensions to use when testing *name*."""
    return REQUIRED_DIM.get(name, DEFAULT_DIM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_FUNCTION_NAMES = get_all_function_names()


@pytest.fixture(scope="session")
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def all_function_names():
    """List of every function name in the package."""
    return ALL_FUNCTION_NAMES
