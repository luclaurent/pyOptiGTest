"""Tests for package-level code: imports, optigtest class, db modules."""


import importlib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

class TestPackageImport:
    """Verify the top-level package can be imported."""

    def test_import_pyOptiGTest(self):
        mod = importlib.import_module("pyOptiGTest")
        assert hasattr(mod, "optigtest")

    def test_import_optigtest_class(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        assert callable(optigtest)

    def test_import_dbProblems(self):
        from pyOptiGTest import dbProblems
        assert hasattr(dbProblems, "loadPb")
        assert hasattr(dbProblems, "listPb")

    def test_import_dbFunctions(self):
        from pyOptiGTest import dbFunctions
        assert dbFunctions is not None

    def test_import_functions_subpackage(self):
        from pyOptiGTest.functions import funAckley2
        assert callable(funAckley2)


# ---------------------------------------------------------------------------
# dbProblems
# ---------------------------------------------------------------------------

class TestDbProblems:
    """Test the problem database module."""

    def test_listPb_returns_dict(self):
        from pyOptiGTest.dbProblems import listPb
        result = listPb()
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_loadPb_known(self):
        from pyOptiGTest.dbProblems import loadPb
        data = loadPb("Ackley2", dim=5)
        assert isinstance(data, dict)
        assert "funobj" in data
        assert "space" in data

    def test_loadPb_unknown_returns_empty(self):
        from pyOptiGTest.dbProblems import loadPb
        data = loadPb("__nonexistent__", dim=5)
        assert isinstance(data, dict)
        assert len(data) == 0

    def test_loadPb_no_name_returns_all(self):
        from pyOptiGTest.dbProblems import loadPb
        data = loadPb()
        assert isinstance(data, dict)
        assert len(data) >= 1


# ---------------------------------------------------------------------------
# optigtest class
# ---------------------------------------------------------------------------

class TestOptigtestClass:
    """Tests for the main optigtest class."""

    def test_create_empty(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        assert obj.namePb == ""
        assert obj.dim == 0

    def test_create_with_name(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Ackley2", dim=5)
        assert obj.namePb == "Ackley2"
        assert obj.dim == 5

    def test_initObj(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        obj.initObj()
        assert obj.namePb == ''
        assert obj.dim == 0

    def test_initEval(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        obj.initEval()
        assert obj.objEval == []
        assert obj.consEval == []

    def test_prepX(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        X = np.array([[1, 2, 3]])
        result = obj.prepX(X)
        np.testing.assert_array_equal(result, X)
        np.testing.assert_array_equal(obj.Xeval, X)

    def test_prepX_scalar_to_2d(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        X = [1, 2, 3]
        result = obj.prepX(X)
        assert result.ndim == 2

    def test_listPb_returns_dict(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        result = obj.listPb()
        assert isinstance(result, dict)

    def test_showDetails(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Ackley2", dim=3)
        # Should not raise
        obj.showDetails(verbose=True)
        obj.showDetails(verbose=False)

    def test_populate_with_dict(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        data = {
            "funobj": ["Ackley1"],
            "funcons": None,
            "type": "Unconstrained",
            "dim": np.inf,
            "space": np.array([-35, 35]),
            "minFglob": 0,
            "minXglob": np.zeros(5),
        }
        obj.populate(data)
        assert obj.funObj == ["Ackley1"]
        assert obj.typePb == "Unconstrained"

    def test_getDesignSpace(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Ackley2", dim=3)
        ds = obj.getDesignSpace()
        assert isinstance(ds, np.ndarray)
        assert ds.ndim == 2

    def test_getXmin_getXmax(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Ackley2", dim=3)
        xmin = obj.getXmin()
        xmax = obj.getXmax()
        assert len(xmin) > 0
        assert len(xmax) > 0
        assert np.all(xmin <= xmax)

    def test_getGlobZmin(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Simionescu")
        z = obj.getGlobZmin()
        assert z == pytest.approx(-0.072, abs=1e-3)

    def test_getNbObj(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest("Ackley2", dim=3)
        assert obj.getNbObj() >= 1

    def test_getNbCons(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest()
        assert obj.getNbCons() == 0


# ---------------------------------------------------------------------------
# Functions __init__ imports
# ---------------------------------------------------------------------------

class TestFunctionsInit:
    """Verify that the functions subpackage exports work."""

    def test_funAckley2_accessible(self):
        from pyOptiGTest.functions import funAckley2
        X = np.random.default_rng(0).random((3, 5))
        result = funAckley2(X)
        assert result.shape == (3,)

    def test_funRosenbrock_accessible(self):
        from pyOptiGTest.functions import funRosenbrock
        X = np.ones((1, 5))
        result = funRosenbrock(X)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# dbConstrained
# ---------------------------------------------------------------------------

class TestDbConstrained:
    """Test the constrained problem database."""

    def test_listPb_returns_dict(self):
        from pyOptiGTest.dbConstrained import listPb
        result = listPb()
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_has_expected_problems(self):
        from pyOptiGTest.dbConstrained import listPb
        result = listPb()
        expected = {'RosenbrockCubicLine', 'RosenbrockDisk', 'BirdDisk',
                    'Townsend', 'Simionescu'}
        assert set(result.keys()) == expected

    def test_loadPb_known(self):
        from pyOptiGTest.dbConstrained import loadPb
        data = loadPb('Simionescu')
        assert isinstance(data, dict)
        assert data['type'] == 'Constrained'
        assert data['dim'] == 2
        assert len(data['funcons']) >= 1

    def test_loadPb_unknown_returns_empty(self):
        from pyOptiGTest.dbConstrained import loadPb
        data = loadPb('__nonexistent__')
        assert isinstance(data, dict)
        assert len(data) == 0

    def test_all_have_required_keys(self):
        from pyOptiGTest.dbConstrained import listPb
        for name, pb in listPb().items():
            for key in ('funobj', 'funcons', 'typecons', 'type', 'dim', 'space'):
                assert key in pb, f'{name} missing key {key}'


# ---------------------------------------------------------------------------
# dbMultiObj
# ---------------------------------------------------------------------------

class TestDbMultiObj:
    """Test the multi-objective problem database."""

    def test_listPb_returns_dict(self):
        from pyOptiGTest.dbMultiObj import listPb
        result = listPb()
        assert isinstance(result, dict)
        assert len(result) == 17

    def test_has_expected_problems(self):
        from pyOptiGTest.dbMultiObj import listPb
        result = listPb()
        expected = {'BinhKorn', 'ChakongHaimes', 'FonsecaFleming', 'TestFun4',
                    'Kursawe', 'MultiSchaffer1', 'MultiSchaffer2', 'Poloni',
                    'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6',
                    'OsyczkaKundu', 'CTP1', 'ConstrEx', 'Viennet'}
        assert set(result.keys()) == expected

    def test_all_multiobj_type(self):
        from pyOptiGTest.dbMultiObj import listPb
        for name, pb in listPb().items():
            assert pb['type'] == 'MultiObjective', f'{name} has wrong type'

    def test_all_have_required_keys(self):
        from pyOptiGTest.dbMultiObj import listPb
        for name, pb in listPb().items():
            for key in ('funobj', 'funcons', 'typecons', 'type', 'dim', 'space'):
                assert key in pb, f'{name} missing key {key}'
            assert len(pb['funobj']) >= 2, f'{name}: multi-obj must have >= 2 objectives'

    def test_loadPb_known(self):
        from pyOptiGTest.dbMultiObj import loadPb
        data = loadPb('ZDT1')
        assert isinstance(data, dict)
        assert data['dim'] == 30

    def test_loadPb_unknown_returns_empty(self):
        from pyOptiGTest.dbMultiObj import loadPb
        data = loadPb('__nonexistent__')
        assert isinstance(data, dict)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# dbProblems aggregation
# ---------------------------------------------------------------------------

class TestDbProblemsAggregation:
    """Test that dbProblems aggregates all three categories."""

    def test_listPb_has_unconstrained(self):
        from pyOptiGTest.dbProblems import listPb
        result = listPb()
        # Should have unconstrained problems (function-based)
        uncons = {k for k, v in result.items() if v.get('type') == 'Unconstrained'}
        assert len(uncons) > 0

    def test_listPb_has_constrained(self):
        from pyOptiGTest.dbProblems import listPb
        result = listPb()
        cons = {k for k, v in result.items() if v.get('type') == 'Constrained'}
        assert len(cons) == 5

    def test_listPb_has_multiobj(self):
        from pyOptiGTest.dbProblems import listPb
        result = listPb()
        multi = {k for k, v in result.items() if v.get('type') == 'MultiObjective'}
        assert len(multi) == 17

    def test_listPbByType(self):
        from pyOptiGTest.dbProblems import listPbByType
        cons = listPbByType('Constrained')
        assert len(cons) == 5
        multi = listPbByType('MultiObjective')
        assert len(multi) == 17


# ---------------------------------------------------------------------------
# optigtest class with constrained/multi-obj
# ---------------------------------------------------------------------------

class TestOptigtestConstrained:
    """Test optigtest class with constrained problems."""

    def test_load_constrained(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('Simionescu')
        assert obj.namePb == 'Simionescu'
        assert obj.typePb == 'Constrained'
        assert obj.getNbObj() == 1
        assert obj.getNbCons() == 1

    def test_eval_constrained(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('Simionescu')
        X = np.array([[0.5, 0.5]])
        result = obj.evalObj(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_eval_constraint(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('Simionescu')
        X = np.array([[0.5, 0.5]])
        result = obj.evalCons(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_check_cons(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('Simionescu')
        X = np.array([[0.0, 0.0]])  # origin should be feasible
        feas = obj.checkCons(X)
        assert isinstance(feas, np.ndarray)
        assert feas.dtype == bool


class TestOptigtestMultiObj:
    """Test optigtest class with multi-objective problems."""

    def test_load_multiobj(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('BinhKorn')
        assert obj.namePb == 'BinhKorn'
        assert obj.typePb == 'MultiObjective'
        assert obj.getNbObj() == 2
        assert obj.getNbCons() == 2

    def test_eval_multiobj(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('BinhKorn')
        X = np.array([[1.0, 1.0]])
        result = obj.evalObj(X)
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, np.ndarray)

    def test_eval_single_obj(self):
        from pyOptiGTest.pyOptiGTest import optigtest
        obj = optigtest('BinhKorn')
        X = np.array([[1.0, 1.0]])
        result = obj.evalObj(X, num=[0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
