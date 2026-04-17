"""
    pyOptiGTest - Database of multi-objective optimization test problems.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import numpy as np


def listPb():
    """Return a dictionary of all multi-objective optimization test problems.

    Each entry maps a problem name to a dict with keys:
        - funobj: list of objective function names
        - funcons: list of constraint function names (or None)
        - typecons: list of constraint types (or None)
        - type: 'MultiObjective'
        - dim: number of design variables (int or np.inf)
        - minFglob: np.nan (Pareto front, no single minimum)
        - minXglob: np.nan (Pareto set)
        - space: design space bounds as numpy array, shape (dim, 2) or list
    """

    pb = {
        'BinhKorn': {
            'funobj': ['funObjKornBinh1', 'funObjKornBinh2'],
            'funcons': ['funConsKornBinh1', 'funConsKornBinh2'],
            'typecons': ['<=', '>='],
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[0.0, 5.0], [0.0, 3.0]]),
        },
        'ChakongHaimes': {
            'funobj': ['funObjChakongHaimes1', 'funObjChakongHaimes2'],
            'funcons': ['funConsChakongHaimes1', 'funConsChakongHaimes2'],
            'typecons': ['<=', '<='],
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-20.0, 20.0], [-20.0, 20.0]]),
        },
        'FonsecaFleming': {
            'funobj': ['funObjFonsecaFleming1', 'funObjFonsecaFleming2'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': np.inf,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-4.0, 4.0]]),
        },
        'TestFun4': {
            'funobj': ['funObjTestFun41', 'funObjTestFun42'],
            'funcons': ['funConsTestFun41', 'funConsTestFun42', 'funConsTestFun43'],
            'typecons': ['>=', '>=', '>='],
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[0.1, 1.0], [-1.0, 1.0]]),
        },
        'Kursawe': {
            'funobj': ['funObjKursawe1', 'funObjKursawe2'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 3,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]),
        },
        'MultiSchaffer1': {
            'funobj': ['funObjMultiSchaffer11', 'funObjMultiSchaffer12'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 1,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-10.0, 10.0]]),
        },
        'MultiSchaffer2': {
            'funobj': ['funObjMultiSchaffer21', 'funObjMultiSchaffer22'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 1,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-5.0, 10.0]]),
        },
        'Poloni': {
            'funobj': ['funObjPoloni1', 'funObjPoloni2'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-np.pi, np.pi], [-np.pi, np.pi]]),
        },
        'ZDT1': {
            'funobj': ['funObjZDT11', 'funObjZDT12'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 30,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.tile([0.0, 1.0], (30, 1)),
        },
        'ZDT2': {
            'funobj': ['funObjZDT21', 'funObjZDT22'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 30,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.tile([0.0, 1.0], (30, 1)),
        },
        'ZDT3': {
            'funobj': ['funObjZDT31', 'funObjZDT32'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 30,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.tile([0.0, 1.0], (30, 1)),
        },
        'ZDT4': {
            'funobj': ['funObjZDT41', 'funObjZDT42'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 10,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.vstack([[0.0, 1.0]] + [[-5.0, 5.0]] * 9),
        },
        'ZDT6': {
            'funobj': ['funObjZDT61', 'funObjZDT62'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 10,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.tile([0.0, 1.0], (10, 1)),
        },
        'OsyczkaKundu': {
            'funobj': ['funObjOsyczkaKundu1', 'funObjOsyczkaKundu2'],
            'funcons': [
                'funConsOsyczkaKundu1', 'funConsOsyczkaKundu2',
                'funConsOsyczkaKundu3', 'funConsOsyczkaKundu4',
                'funConsOsyczkaKundu5', 'funConsOsyczkaKundu6',
            ],
            'typecons': ['>=', '>=', '>=', '>=', '>=', '>='],
            'type': 'MultiObjective',
            'dim': 6,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([
                [0.0, 10.0], [0.0, 10.0], [1.0, 5.0],
                [0.0, 6.0], [1.0, 5.0], [0.0, 10.0],
            ]),
        },
        'CTP1': {
            'funobj': ['funObjCTP11', 'funObjCTP12'],
            'funcons': ['funConsCTP11', 'funConsCTP12'],
            'typecons': ['>=', '>='],
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[0.0, 1.0], [0.0, 1.0]]),
        },
        'ConstrEx': {
            'funobj': ['funObjConstrEx1', 'funObjConstrEx2'],
            'funcons': ['funConsConstrEx1', 'funConsConstrEx2'],
            'typecons': ['>=', '>='],
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[0.1, 1.0], [0.0, 5.0]]),
        },
        'Viennet': {
            'funobj': ['funObjViennet1', 'funObjViennet2', 'funObjViennet3'],
            'funcons': None,
            'typecons': None,
            'type': 'MultiObjective',
            'dim': 2,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-3.0, 3.0], [-3.0, 3.0]]),
        },
    }
    return pb


def loadPb(pbName=None):
    """Load multi-objective problem(s) by name.

    Args:
        pbName: Problem name string. If None, return all problems.

    Returns:
        dict: Single problem dict if pbName given, else all problems.
    """

    allPb = listPb()
    if pbName:
        return allPb.get(pbName, {})
    return allPb
