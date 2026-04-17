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


class FunCosineMixture(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        design space -1<xi<1
        """

        #constants
        a=0.1
        b=5*np.pi

        #evaluation and derivatives
        pa=np.cos(b*X)
        pb=X**2
        p=a*np.sum(pa, axis=1)+np.sum(pb, axis=1)
        if grad:
            dp=-a*b*np.sin(b*X)+2*X

        return (p, dp) if grad else p


funCosineMixture = FunCosineMixture()
