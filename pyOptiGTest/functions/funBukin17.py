"""
    pyOptiGTest - Python implementation of optimization test functions.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import numpy as np

from pyOptiGTest.base import TestFunction


class FunBukin17(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 17's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=2
        c=10
        d=3


        #evaluation and derivatives
        pa=X[:, 1]+b*X[:, 0]-c
        pb=d*X[:, 1]-X[:, 0]+c
        pc=d*X[:, 0]-X[:, 1]+c
        pd=X[:, 0]+X[:, 1]+c
        p=a*np.abs(pa*pb*pc)+np.abs(pd)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=a*(b*pb*pc-pa*pc+d*pa*pb)*np.sign(pa*pb*pc)+np.sign(pd)
            dp[:, 1]=a*(pb*pc+d*pa*pc-pa*pb)*np.sign(pa*pb*pc)+np.sign(pd)

        return (p, dp) if grad else p


funBukin17 = FunBukin17()
