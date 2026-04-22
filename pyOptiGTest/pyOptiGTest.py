"""
    pyOptiGTest - OptiGTest class for managing optimization test problems.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""



import logging
import importlib
import sys
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from . import dbProblems as dbP

textSpacer: str = '=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#='
smalltextSpacer: str = '------------------------'


class optigtest:
    """Main class for managing and evaluating optimization test problems.

    Supports unconstrained, constrained, and multi-objective problems.

    Args:
        namePb: Name of the test problem (optional).
        X: Sample points to evaluate, shape (N, dim) (optional).
        dim: Problem dimension (optional, required for variable-dim problems).
    """

    def __init__(
        self,
        namePb: str | None = None,
        X: npt.ArrayLike | None = None,
        dim: int | None = None,
    ) -> None:
        # initialize object
        self.initObj()
        # initialize logging
        logging.basicConfig(
            handlers=[logging.StreamHandler(sys.stdout)],
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logging.info('###############################')
        logging.info('### Create optiGTest object ###')
        logging.info('###############################')

        if namePb:
            self.setPbName(namePb)
        if dim:
            self.setDim(dim)
        if namePb and X is None:
            self.showDetails()
        if X is not None:
            self.prepX(X)
            self.evalAll()

    def initObj(self) -> None:
        """Initialize the structure of the object."""
        self.namePb: str = ''
        self.typePb: str = ''
        self.dim: int = 0
        self.Xeval: npt.NDArray[np.floating[Any]] | None = None
        self.populate()
        self.initEval()

    def initEval(self) -> None:
        """Initialize storage for evaluations."""
        self.objEval: list[Any] = []
        self.consEval: list[Any] = []
        self.objGradEval: list[Any] = []
        self.consGradEval: list[Any] = []

    def populate(self, dictPb: dict[str, Any] | None = None) -> None:
        """Populate object attributes from a problem dictionary."""
        self.typePb: str = ''
        self.funObj: list[str] = []
        self.funCons: list[str] = []
        self.typeCons: list[str] = []
        self.designSpace: npt.NDArray[np.floating[Any]] = np.empty((0, 2))
        self.dimAvailable: int | float = 0
        self.locMinZ: list[Any] = []
        self.locMinX: list[Any] = []
        self.globMinZ: float | npt.NDArray[np.floating[Any]] = np.nan
        self.globMinX: float | npt.NDArray[np.floating[Any]] = np.nan

        if dictPb is not None:
            self.funObj = dictPb.get('funobj', []) or []
            self.funCons = dictPb.get('funcons', []) or []
            self.typePb = dictPb.get('type', '')
            self.typeCons = dictPb.get('typecons', []) or []
            self.dimAvailable = dictPb.get('dim', 0)
            if 'space' in dictPb:
                self.designSpace = np.atleast_2d(dictPb['space'])
            if 'minFglob' in dictPb:
                self.globMinZ = dictPb['minFglob']
            if 'minXglob' in dictPb:
                self.globMinX = dictPb['minXglob']
            if 'minFloc' in dictPb:
                self.locMinZ = dictPb['minFloc']
            if 'minXloc' in dictPb:
                self.locMinX = dictPb['minXloc']

    def loadData(self, pbName: str | None = None, dim: int | None = None) -> dict[str, Any]:
        """Load data of the selected test case."""
        dataPb = dbP.loadPb(pbName, self.dim)
        self.populate()

        if pbName:
            if len(dataPb) == 0:
                logging.error('Problem {} not available'.format(pbName))
                dataPb = dbP.loadPb()
                self.showPbs(dataPb)
            else:
                currentDim = dim if dim is not None else self.dim
                status = self.dimensionOk(currentDim, dataPb['dim'])
                if status != -1:
                    self.populate(dataPb)
                else:
                    logging.error('Dimension {} not available for problem {}'.format(
                        currentDim, pbName))
        else:
            self.showPbs(dataPb)
        return dataPb

    # ---- Accessors ----

    def listPb(self, pbType: str | None = None) -> dict[str, dict[str, Any]]:
        """Return the list of all available problems, optionally filtered by type."""
        allPb: dict[str, dict[str, Any]] = dbP.listPbByType(pbType, self.dim)
        for pbname in sorted(allPb.keys()):
            logging.info('{} ({})'.format(pbname, allPb[pbname].get('type', '')))
        return allPb

    def prepX(self, X: npt.ArrayLike | None = None) -> npt.NDArray[np.floating[Any]] | None:
        """Pack points into (N, dim) array."""
        if X is not None:
            self.Xeval = np.atleast_2d(X)
        return self.Xeval

    def getDesignSpace(self) -> npt.NDArray[np.floating[Any]]:
        """Return the design space, tiled if needed for the current dimension."""
        dS = self.designSpace
        if dS.shape[0] == 1 and self.dim > 1:
            dS = np.tile(dS, (self.dim, 1))
        return dS

    def getXmax(self) -> npt.NDArray[np.floating[Any]]:
        return self.getDesignSpace()[:, 1]

    def getXmin(self) -> npt.NDArray[np.floating[Any]]:
        return self.getDesignSpace()[:, 0]

    def getGlobZmin(self) -> float | npt.NDArray[np.floating[Any]]:
        return self.globMinZ

    def getGlobXmin(self) -> float | npt.NDArray[np.floating[Any]]:
        return self.globMinX

    def getTypePb(self) -> str:
        return self.typePb

    def getNbObj(self) -> int:
        return len(self.funObj) if isinstance(self.funObj, list) else 1

    def getNbCons(self) -> int:
        if self.funCons is None:
            return 0
        return len(self.funCons) if isinstance(self.funCons, list) else 1

    @staticmethod
    def dimensionOk(dimToCheck: int | float, dimAvailable: int | float | npt.ArrayLike) -> int:
        """Check if the chosen dimension is valid.

        Returns:
            -1: bad dimension
             1: dimension is allowed
             n (n>0): no selected dimension, only one is available
        """
        status = -1
        if dimAvailable == np.inf:
            status = 1
        else:
            dimAvailable = np.atleast_1d(dimAvailable)
            if dimToCheck in dimAvailable:
                status = 1
            elif len(dimAvailable) == 1:
                status = int(dimAvailable[0])
        return status

    def setPbName(self, pbName: str | None = None) -> None:
        """Declare the problem and load data."""
        if pbName and self.loadData(pbName):
            self.namePb = pbName

    def setDim(self, dim: int | None = None) -> None:
        """Set the problem dimension and reload data."""
        if dim is not None:
            self.dim = dim
            self.loadData(pbName=self.namePb)

    # ---- Function loading ----

    @staticmethod
    def _load_function(funName: str) -> Callable[..., Any]:
        """Load a function by name from the functions subpackage."""
        from pyOptiGTest.base import TestFunction
        try:
            return TestFunction.from_name(funName)
        except (AttributeError, TypeError):
            # Fallback for plain-function modules (no class)
            mod = importlib.import_module('pyOptiGTest.functions.{}'.format(funName))
            return getattr(mod, funName)

    # ---- Evaluation ----

    def evalAll(self, X: npt.ArrayLike | None = None) -> None:
        """Evaluate all objective and constraint functions."""
        self.objEval = self.evalObj(X)
        if self.funCons:
            self.consEval = self.evalCons(X)

    def evalObj(self, X: npt.ArrayLike | None = None, grad: bool = False, num: list[int] | None = None) -> Any:
        """Evaluate objective function(s).

        Args:
            X: Sample points, shape (N, dim). If None, use self.Xeval.
            grad: Whether to also compute gradients.
            num: List of objective indices to evaluate (default: all).

        Returns:
            list or single result: Function evaluations (and gradients if requested).
        """
        Xrun: npt.NDArray[np.floating[Any]] | None = self.Xeval
        if X is not None:
            Xrun = self.prepX(X)

        numOK: list[int] = list(range(len(self.funObj)))
        if num is not None:
            numOK = list(np.atleast_1d(num))

        results: list[Any] = []
        for iF in numOK:
            fn: Callable[..., Any] = self._load_function(self.funObj[iF])
            results.append(fn(Xrun, grad=grad))

        if len(numOK) == 1:
            return results[0]
        return results

    def evalCons(self, X: npt.ArrayLike | None = None, grad: bool = False, num: list[int] | None = None) -> Any:
        """Evaluate constraint function(s).

        Args:
            X: Sample points, shape (N, dim). If None, use self.Xeval.
            grad: Whether to also compute gradients.
            num: List of constraint indices to evaluate (default: all).

        Returns:
            list or single result: Constraint evaluations.
        """
        if not self.funCons:
            return []

        Xrun: npt.NDArray[np.floating[Any]] | None = self.Xeval
        if X is not None:
            Xrun = self.prepX(X)

        numOK: list[int] = list(range(len(self.funCons)))
        if num is not None:
            numOK = list(np.atleast_1d(num))

        results: list[Any] = []
        for iC in numOK:
            fn: Callable[..., Any] = self._load_function(self.funCons[iC])
            results.append(fn(Xrun, grad=grad))

        if len(numOK) == 1:
            return results[0]
        return results

    def checkCons(self, X: npt.ArrayLike | None = None, Z: list[npt.NDArray[np.floating[Any]]] | None = None) -> npt.NDArray[np.bool_] | None:
        """Check constraint feasibility.

        Returns:
            numpy array of booleans, True if all constraints are satisfied.
        """
        if not self.funCons:
            return None

        if Z is None:
            Z = self.evalCons(X)

        if not isinstance(Z, list):
            Z = [Z]

        feasible: npt.NDArray[np.bool_] = np.ones(Z[0].shape[0], dtype=bool)
        for iC, (z, tcons) in enumerate(zip(Z, self.typeCons)):
            if tcons == '<=':
                feasible &= (z.flatten() <= 0)
            elif tcons == '<':
                feasible &= (z.flatten() < 0)
            elif tcons == '>=':
                feasible &= (z.flatten() >= 0)
            elif tcons == '>':
                feasible &= (z.flatten() > 0)
            elif tcons == '=':
                feasible &= np.isclose(z.flatten(), 0)
        return feasible

    # ---- Display ----

    def showDetails(self, verbose: bool = True) -> None:
        """Show details of the current problem."""
        if verbose:
            logging.info(textSpacer)
        logging.info('Problem: {}'.format(self.namePb))
        logging.info('Type: {}'.format(self.typePb))
        logging.info('Dimension: {}'.format(self.dimAvailable))
        logging.info('Nb objectives: {}'.format(self.getNbObj()))
        logging.info('Nb constraints: {}'.format(self.getNbCons()))
        if not np.all(np.isnan(np.atleast_1d(self.globMinZ))):
            logging.info('Global minimum: {}'.format(self.globMinZ))
        if verbose:
            logging.info(textSpacer)

    def showPbs(self, dictPb: dict[str, dict[str, Any]] | None = None) -> None:
        """Show available problems grouped by type."""
        if dictPb is None:
            return

        dictUn: dict[str, dict[str, Any]] = {k: v for k, v in dictPb.items() if v.get('type') == 'Unconstrained'}
        dictCons: dict[str, dict[str, Any]] = {k: v for k, v in dictPb.items() if v.get('type') == 'Constrained'}
        dictMulti: dict[str, dict[str, Any]] = {k: v for k, v in dictPb.items() if v.get('type') == 'MultiObjective'}
        nbPb: int = len(dictUn) + len(dictCons) + len(dictMulti)

        logging.info(textSpacer)
        if nbPb == 0:
            logging.info('No available test problems')
        else:
            logging.info('{} available test problems'.format(nbPb))
            if dictUn:
                logging.info(smalltextSpacer)
                logging.info('Unconstrained ({})'.format(len(dictUn)))
                for x in sorted(dictUn.keys()):
                    logging.info('  {}'.format(x))
            if dictCons:
                logging.info(smalltextSpacer)
                logging.info('Constrained ({})'.format(len(dictCons)))
                for x in sorted(dictCons.keys()):
                    nb_cons = len(dictCons[x].get('funcons', []) or [])
                    logging.info('  {} (constraints: {})'.format(x, nb_cons))
            if dictMulti:
                logging.info(smalltextSpacer)
                logging.info('Multi-objective ({})'.format(len(dictMulti)))
                for x in sorted(dictMulti.keys()):
                    nb_obj = len(dictMulti[x].get('funobj', []) or [])
                    nb_cons = len(dictMulti[x].get('funcons', []) or [])
                    logging.info('  {} (objectives: {}, constraints: {})'.format(
                        x, nb_obj, nb_cons))
        logging.info(textSpacer)
