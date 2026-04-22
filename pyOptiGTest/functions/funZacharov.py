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


class FunZacharov(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Zacharov function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -5.12<xi<5.12
        """

        #constants
        a=1/2

        #evaluation and derivatives
        sX=X.shape[1]
        #
        li=np.arange(1, sX+1)
        pa=X**2
        pb=li*X
        pbb=np.sum(pb, axis=1)
        #
        p=np.sum(pa, axis=1)+a**2*pbb**2+a**4*pbb**4

        if grad:
            dp=2*X+2*a**2*li*pbb+4*a**4*li*pbb**3

        return (p, dp) if grad else p


funZacharov = FunZacharov()
