# pyOptiGTest

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python library of **282+ classical optimization test functions** for benchmarking optimization and metaheuristic algorithms. Includes unconstrained, constrained, and multi-objective problems. All functions support vectorized NumPy evaluation and analytical gradient computation.

## Features

- **282 test functions** — Ackley, Beale, Branin, Rosenbrock, Rastrigin, Griewank, Schwefel, Sphere, and many more
- **Analytical gradients** — every function provides exact gradient computation via the `grad` parameter
- **Vectorized evaluation** — efficient batch evaluation using NumPy arrays
- **Known global optima** — each problem includes reference global minimum values and locations
- **Variable dimensionality** — many functions support arbitrary dimensions; fixed-dimension functions are also included
- **Problem types** — unconstrained, constrained, and multi-objective problems
- **Lightweight** — only depends on NumPy

## Installation

```bash
pip install pyOptiGTest
```

Or install from source:

```bash
git clone https://github.com/luclaurent/optigtest.git
cd optigtest
pip install .
```

For testing:

```bash
pip install -e ".[test]"
```
and run
```bash
pytest 
```
or 
```bash
pytest --pyargs pyOptiGTest
```


## Quick Start

### Using the `optigtest` class

```python
import pyOptiGTest as PO
import numpy as np

# Load a test problem
obj = PO.optigtest('Ackley1', dim=2)

# Get design space bounds
xmin = obj.getXmin()
xmax = obj.getXmax()

# Create evaluation points
X = np.random.uniform(xmin, xmax, size=(100, 2))

# Evaluate the objective function
Z = obj.evalObj(X)

# Access known global minimum
print("Global minimum value:", obj.getGlobZmin())
print("Global minimum location:", obj.getGlobXmin())
```

### Calling functions directly

Each function can also be called directly from the `functions` submodule:

```python
from pyOptiGTest.functions.funRosenbrock import funRosenbrock
import numpy as np

X = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 3.0]])

# Function values only
Z = funRosenbrock(X)

# Function values and gradients
Z, dZ = funRosenbrock(X, grad=True)
# Z shape:  (3,)
# dZ shape: (3, 2)
```

### Visualizing a test function

```python
import pyOptiGTest as PO
import numpy as np
import matplotlib.pyplot as plt

obj = PO.optigtest('Ackley1', dim=2)

# Build evaluation grid
x = np.linspace(obj.getXmin()[0], obj.getXmax()[0], 200)
y = np.linspace(obj.getXmin()[1], obj.getXmax()[1], 200)
Xm, Ym = np.meshgrid(x, y)
XX = np.column_stack([Xm.ravel(), Ym.ravel()])

Z = obj.evalObj(XX)
Zm = Z.reshape(Xm.shape)

# Contour plot
plt.contourf(Xm, Ym, Zm, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Ackley1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
```

## Function Interface

All test functions follow a consistent interface:

```python
def funName(X, grad=False):
    """
    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape (n_samples, n_dimensions).
    grad : bool, optional
        If True, also compute and return analytical gradients.

    Returns
    -------
    p : numpy.ndarray
        Function values, shape (n_samples,).
    dp : numpy.ndarray (only if grad=True)
        Gradient array, shape (n_samples, n_dimensions).
    """
```

## `optigtest` Class API

| Method | Description |
|---|---|
| `optigtest(namePb, dim=None, X=None)` | Create a problem instance |
| `evalObj(X, grad=False, num=None)` | Evaluate objective function(s) at points `X` |
| `evalCons(X, grad=False, num=None)` | Evaluate constraint function(s) at points `X` |
| `checkCons(X)` | Check constraint feasibility (returns boolean array) |
| `getDesignSpace()` | Get bounds array of shape `(dim, 2)` |
| `getXmin()` / `getXmax()` | Get lower / upper bounds |
| `getGlobZmin()` / `getGlobXmin()` | Get global minimum value / location |
| `getNbObj()` / `getNbCons()` | Get number of objectives / constraints |
| `getTypePb()` | Get problem type (`Unconstrained`, `Constrained`, `MultiObjective`) |
| `setDim(dim)` | Change problem dimensionality |
| `listPb(pbType=None)` | List all available problems (optionally filtered by type) |
| `showDetails()` | Print problem details |

## Constrained Problems

5 constrained test problems with analytical constraints:

| Problem | Constraints | Known Minimum |
|---|---|---|
| RosenbrockCubicLine | 2 (<=) | 0.0 |
| RosenbrockDisk | 1 (<=) | 0.0 |
| BirdDisk | 1 (<) | -106.76 |
| Townsend | 1 (<) | -2.024 |
| Simionescu | 1 (<=) | -0.072 |

```python
from pyOptiGTest.pyOptiGTest import optigtest
import numpy as np

pb = optigtest("Simionescu")
X = np.array([[0.5, 0.5]])
obj = pb.evalObj(X)
cons = pb.evalCons(X)
feasible = pb.checkCons(X)
```

## Multi-Objective Problems

17 multi-objective test problems (BinhKorn, ChakongHaimes, FonsecaFleming, Kursawe, ZDT1–6, Viennet, etc.):

```python
from pyOptiGTest.pyOptiGTest import optigtest
import numpy as np

pb = optigtest("BinhKorn")
X = np.array([[1.0, 1.0]])
results = pb.evalObj(X)  # returns [f1, f2]
feasible = pb.checkCons(X)
```

See [examples.md](examples.md) for detailed usage examples.

## Available Functions (selection)

Below is a non-exhaustive list of included test functions (almost complete list can be found [here](wiki/index.md)):

| Function | Dimensions | Global Minimum |
|---|---|---|
| Ackley (1–4) | any | 0 |
| Alpine (1–2) | any | 0 |
| Beale | 2 | 0 |
| Booth | 2 | 0 |
| Branin (1–2) | 2 | 0 |
| Bukin (01–20) | 2 | varies |
| Camelback (3-hump, 6-hump) | 2 | 0 |
| Colville | 4 | 0 |
| Cross-in-Tray | 2 | −2.0626 |
| Dixon-Price | any | 0 |
| Drop-Wave | 2 | −1 |
| Easom | 2 | −1 |
| Egg Holder | 2 | −959.6407 |
| Goldstein-Price | 2 | 3 |
| Griewank | any | 0 |
| Hartmann (3, 6) | 3, 6 | varies |
| Levy | any | 0 |
| Michalewicz | any | varies |
| Rastrigin | any | 0 |
| Rosenbrock | any | 0 |
| Schwefel | any | 0 |
| Sphere | any | 0 |
| Styblinski-Tang | any | varies |
| ... and 250+ more | | |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=pyOptiGTest --cov-report=term-missing tests/
```

## Project Structure

```
pyOptiGTest/
├── pyOptiGTest/
│   ├── __init__.py
│   ├── pyOptiGTest.py        # Main optigtest class
│   ├── dbProblems.py          # Problem database
│   ├── dbFunctions.py         # Function registry
│   ├── dbConstrained.py       # Constrained problem definitions
│   ├── dbUnconstrained.py     # Unconstrained problem definitions
│   ├── dbMultiObj.py          # Multi-objective problem definitions
│   └── functions/             # 282 individual test functions
│       ├── funAckley2.py
│       ├── funBeale.py
│       ├── funRosenbrock.py
│       └── ...
├── tests/                     # Test suite
├── pyproject.toml
├── LICENSE
└── README.md
```

## MATLAB/Octave version

A similar MATLAB/Octave toolbox can be found [here](https://github.com/luclaurent/optiGTest).

## References

This toolbox is inspired by many existing codes and papers

* [AMPGO](http://infinity77.net/global_optimization/index.html) (and [github repository](https://github.com/andyfaff/ampgo/))
* E. P. Adorio and U. P. Diliman. [MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization.](https://www.google.fr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi2j_iz8sbTAhWG0hoKHfYLAncQFggnMAA&url=http%3A%2F%2Fwww.geocities.ws%2Feadorio%2Fmvf.pdf&usg=AFQjCNE7AMN9NpxLz2UGDInWKcwMeC120g&sig2=trbG1un24A4RfYCPdifjuA)
* P. N. Suganthan, N. Hansen, J. J. Liang, K. Deb, Y. P. Chen, A. Auger and S. Tiwari. Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter optimization. KanGAL report, 2005.[Link](https://www.lri.fr/~hansen/Tech-Report-May-30-05.pdf)
* V. Bicik, Continuous optimization algorithms, Master thesis, Czech Technical University in Prague, 2010 [Link](https://dip.felk.cvut.cz/browse/pdfcache/bicikvla_2010dipl.pdf)
* M. Jamil and Xin-She Yang, A literature survey of benchmark functions for global optimization problems, *Int. Journal of Mathematical Modelling and Numerical Optimisation*, Vol. 4, No. 2, pp. 150--194 (2013) doi: [10.1504/IJMMNO.2013.055204](https://doi.org/10.1504/IJMMNO.2013.055204) arXiv: [1308.4008](https://arxiv.org/abs/1308.4008) [PDF](https://arxiv.org/pdf/1308.4008.pdf)
* M. Molga, C. Smutnick. Test functions for optimization needs, *Comput. Inform. Sci.*, 1-43, 2005. [Link](http://new.zsd.iiar.pwr.wroc.pl/files/docs/functions.pdf)
* M. M. Ali, C. Khompatraporn and Z. B. Zabinsky, *Journal of Global Optimisation* (2005) 31:635. doi: [10.1007/s10898-004-9972-2](https://doi.org/10.1007/s10898-004-9972-2) [PDF](http://folk.uib.no/ssu029/Pdf_file/Ali05.pdf)
* [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/other.html)
* Wikipedia. [Test functions for optimization](http://en.wikipedia.org/w/index.php?title=Test%20functions%20for%20optimization&oldid=837180368), 2018. [Online; accessed 13-May-2018]

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Luc Laurent** — [luc.laurent@lecnam.net](mailto:luc.laurent@lecnam.net)

## Links

- **Repository**: [https://github.com/luclaurent/pyoptigtest](https://github.com/luclaurent/pyoptigtest)
- **Issues**: [https://github.com/luclaurent/pyoptigtest/issues](https://github.com/luclaurent/pyoptigtest/issues)