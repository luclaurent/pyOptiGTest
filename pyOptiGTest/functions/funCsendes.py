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


class FunCsendes(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Csendes's function
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for (x)=(0,...,0)
        Design space: -1<xi<1
        """

        #constants
        a=2

        #evaluation and derivatives
        sx=np.sin(1/X)
        pa=a+sx
        pSum=X**6*pa
        p=np.sum(pSum, axis=1)
        #
        if grad:
            dp=6*X**5*pa-X**4*np.cos(1/X)

        return (p, dp) if grad else p


funCsendes = FunCsendes()
