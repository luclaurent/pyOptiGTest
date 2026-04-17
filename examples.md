# Examples

## Unconstrained Optimization

### Direct function evaluation

```python
import numpy as np
from pyOptiGTest.functions.funRosenbrock import funRosenbrock

# Create sample points: 10 points in 5 dimensions
X = np.random.rand(10, 5)

# Evaluate function values
f = funRosenbrock(X)
print(f"Function values shape: {f.shape}")  # (10,)

# Evaluate with gradients
f, df = funRosenbrock(X, grad=True)
print(f"Gradient shape: {df.shape}")  # (10, 5)
```

### Using the OptiGTest class

```python
from pyOptiGTest.pyOptiGTest import optigtest
import numpy as np

# Create an unconstrained problem
pb = optigtest("Ackley2", dim=5)

# Access problem metadata
print(f"Type: {pb.getTypePb()}")
print(f"Design space bounds:\n{pb.getDesignSpace()}")

# Generate sample points and evaluate
X = np.random.rand(20, 5) * 10 - 5
result = pb.evalObj(X)
print(f"Evaluations: {result.shape}")
```

### Listing available problems

```python
from pyOptiGTest.pyOptiGTest import optigtest

pb = optigtest()

# List all problems
all_problems = pb.listPb()

# List only unconstrained problems
unconstrained = pb.listPb(pbType="Unconstrained")
```

---

## Constrained Optimization

### Direct function evaluation

```python
import numpy as np
from pyOptiGTest.functions.funSimionescu import funSimionescu
from pyOptiGTest.functions.funConsSimionescu import funConsSimionescu

X = np.array([[0.5, 0.5], [-0.3, 0.7], [0.8, -0.8]])

# Evaluate objective
f = funSimionescu(X)
print(f"Objective: {f}")

# Evaluate constraint (feasible when <= 0)
g = funConsSimionescu(X)
print(f"Constraint: {g}")
print(f"Feasible: {g <= 0}")
```

### Using the OptiGTest class

```python
from pyOptiGTest.pyOptiGTest import optigtest
import numpy as np

# Load a constrained problem
pb = optigtest("Simionescu")

print(f"Type: {pb.getTypePb()}")
print(f"Nb objectives: {pb.getNbObj()}")
print(f"Nb constraints: {pb.getNbCons()}")
print(f"Known minimum: {pb.getGlobZmin()}")

# Evaluate at sample points
X = np.array([[0.5, 0.5], [0.0, 0.0]])
obj = pb.evalObj(X)
cons = pb.evalCons(X)
feasible = pb.checkCons(X)

print(f"Objective values: {obj}")
print(f"Constraint values: {cons}")
print(f"Feasibility: {feasible}")
```

### Available constrained problems

| Problem | Dim | Constraints | Known Minimum |
|---------|-----|-------------|---------------|
| RosenbrockCubicLine | 2 | 2 (<=) | 0.0 |
| RosenbrockDisk | 2 | 1 (<=) | 0.0 |
| BirdDisk | 2 | 1 (<) | -106.764537 |
| Townsend | 2 | 1 (<) | -2.0239884 |
| Simionescu | 2 | 1 (<=) | -0.072 |

---

## Multi-Objective Optimization

### Direct function evaluation

```python
import numpy as np
from pyOptiGTest.functions.funObjKornBinh1 import funObjKornBinh1
from pyOptiGTest.functions.funObjKornBinh2 import funObjKornBinh2
from pyOptiGTest.functions.funConsKornBinh1 import funConsKornBinh1
from pyOptiGTest.functions.funConsKornBinh2 import funConsKornBinh2

X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.5]])

# Evaluate both objectives
f1 = funObjKornBinh1(X)
f2 = funObjKornBinh2(X)
print(f"Objective 1: {f1}")
print(f"Objective 2: {f2}")

# Evaluate constraints
g1 = funConsKornBinh1(X)  # feasible when <= 0
g2 = funConsKornBinh2(X)  # feasible when >= 0
print(f"Constraint 1: {g1}")
print(f"Constraint 2: {g2}")
```

### Using the OptiGTest class

```python
from pyOptiGTest.pyOptiGTest import optigtest
import numpy as np

# Load a multi-objective problem
pb = optigtest("BinhKorn")

print(f"Type: {pb.getTypePb()}")
print(f"Nb objectives: {pb.getNbObj()}")
print(f"Nb constraints: {pb.getNbCons()}")
print(f"Design space:\n{pb.getDesignSpace()}")

# Evaluate all objectives
X = np.array([[1.0, 1.0], [2.5, 1.5]])
results = pb.evalObj(X)  # returns list of arrays for multi-obj
for i, r in enumerate(results):
    print(f"Objective {i+1}: {r}")

# Evaluate a single objective
f1 = pb.evalObj(X, num=[0])
print(f"Objective 1 only: {f1}")

# Check constraint feasibility
feasible = pb.checkCons(X)
print(f"Feasibility: {feasible}")
```

### Available multi-objective problems

| Problem | Dim | Objectives | Constraints |
|---------|-----|------------|-------------|
| BinhKorn | 2 | 2 | 2 |
| ChakongHaimes | 2 | 2 | 2 |
| FonsecaFleming | n | 2 | 0 |
| TestFun4 | 2 | 2 | 3 |
| Kursawe | 3 | 2 | 0 |
| MultiSchaffer1 | 1 | 2 | 0 |
| MultiSchaffer2 | 1 | 2 | 0 |
| Poloni | 2 | 2 | 0 |
| ZDT1 | 30 | 2 | 0 |
| ZDT2 | 30 | 2 | 0 |
| ZDT3 | 30 | 2 | 0 |
| ZDT4 | 10 | 2 | 0 |
| ZDT6 | 10 | 2 | 0 |
| OsyczkaKundu | 6 | 2 | 6 |
| CTP1 | 2 | 2 | 2 |
| ConstrEx | 2 | 2 | 2 |
| Viennet | 2 | 3 | 0 |

---

## Complete Optimization with `scipy.optimize`

### Unconstrained: Rosenbrock (scipy.optimize.minimize)

```python
from pyOptiGTest.pyOptiGTest import optigtest
from scipy.optimize import minimize
import numpy as np

pb = optigtest("Rosenbrock", dim=5)

def objective(x):
    X = x.reshape(1, -1)
    f, g = pb.evalObj(X, grad=True)
    return float(f[0]), g[0].astype(float)

x0 = np.zeros(5)
result = minimize(objective, x0, method="L-BFGS-B", jac=True,
                  bounds=list(zip(pb.getXmin(), pb.getXmax())))

print(f"Optimum found: f = {result.fun:.6e}")
print(f"At x = {result.x}")
print(f"Known global min: {pb.getGlobZmin()}")
# Optimum found: f ≈ 0  at x ≈ [1, 1, 1, 1, 1]
```

### Constrained: RosenbrockDisk (scipy.optimize.minimize with SLSQP)

```python
from pyOptiGTest.pyOptiGTest import optigtest
from scipy.optimize import minimize
import numpy as np

pb = optigtest("RosenbrockDisk")

def objective(x):
    X = x.reshape(1, -1)
    f, g = pb.evalObj(X, grad=True)
    return float(f[0]), g[0].astype(float)

def constraint(x):
    """g(x) <= 0, so scipy ineq constraint needs g(x) >= 0 → return -g(x)."""
    X = x.reshape(1, -1)
    c = pb.evalCons(X)
    return -float(c[0])

x0 = np.array([0.5, 0.5])
result = minimize(objective, x0, method="SLSQP", jac=True,
                  bounds=list(zip(pb.getXmin(), pb.getXmax())),
                  constraints={"type": "ineq", "fun": constraint})

print(f"Optimum found: f = {result.fun:.6e}")
print(f"At x = {result.x}")
print(f"Known global min: {pb.getGlobZmin()}")
# Optimum found: f ≈ 0  at x ≈ [1, 1]
```

### Multi-Objective: BinhKorn (weighted-sum scalarization)

For multi-objective problems, a simple approach is to scalarize into a single
objective via a weighted sum, then solve a series of problems with different
weight vectors to approximate the Pareto front.

```python
from pyOptiGTest.pyOptiGTest import optigtest
from scipy.optimize import minimize
import numpy as np

pb = optigtest("BinhKorn")

def make_scalar_obj(weights):
    """Return a scalar objective f(x) = w1*f1(x) + w2*f2(x)."""
    def objective(x):
        X = x.reshape(1, -1)
        f1, g1 = pb.evalObj(X, grad=True, num=[0])
        f2, g2 = pb.evalObj(X, grad=True, num=[1])
        f = weights[0] * float(f1[0]) + weights[1] * float(f2[0])
        g = weights[0] * g1[0] + weights[1] * g2[0]
        return f, g.astype(float)
    return objective

# BinhKorn constraints: cons1 <= 0, cons2 >= 0
def constraint_1(x):
    X = x.reshape(1, -1)
    c = pb.evalCons(X, num=[0])  # <= 0  →  scipy ineq wants >= 0
    return -float(c[0])

def constraint_2(x):
    X = x.reshape(1, -1)
    c = pb.evalCons(X, num=[1])  # >= 0  →  scipy ineq wants >= 0
    return float(c[0])

bounds = list(zip(pb.getXmin(), pb.getXmax()))
constraints = [
    {"type": "ineq", "fun": constraint_1},
    {"type": "ineq", "fun": constraint_2},
]

pareto_points = []
for w1 in np.linspace(0.01, 0.99, 20):
    obj = make_scalar_obj([w1, 1 - w1])
    x0 = np.array([2.5, 1.5])
    res = minimize(obj, x0, method="SLSQP", jac=True,
                   bounds=bounds, constraints=constraints)
    if res.success:
        X = res.x.reshape(1, -1)
        f1 = float(pb.evalObj(X, num=[0])[0])
        f2 = float(pb.evalObj(X, num=[1])[0])
        pareto_points.append((f1, f2))

pareto_points = np.array(pareto_points)
print(f"Approximated {len(pareto_points)} Pareto-optimal points")
# Can be plotted: plt.scatter(pareto_points[:, 0], pareto_points[:, 1])
```

---

## Database Queries

```python
from pyOptiGTest import dbProblems, dbConstrained, dbMultiObj

# Get all problems
all_pbs = dbProblems.listPb()
print(f"Total problems: {len(all_pbs)}")

# Get only constrained problems
cons_pbs = dbConstrained.listPb()
print(f"Constrained: {list(cons_pbs.keys())}")

# Get only multi-objective problems
multi_pbs = dbMultiObj.listPb()
print(f"Multi-objective: {list(multi_pbs.keys())}")

# Filter by type
from pyOptiGTest.dbProblems import listPbByType
constrained = listPbByType("Constrained")
multiobj = listPbByType("MultiObjective")
```
