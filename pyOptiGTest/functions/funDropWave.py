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


class FunDropWave(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-1 for xi=0
        Design space: -5.12<xi<5.12
        """
        #constants
        a=1
        b=12
        c=2
        d=1/2

        #evaluation and derivatives
        sxx=np.sum(X**2, axis=1)
        pa=a+np.cos(b*sxx)
        pb=c+d*sxx
        #
        p=-pa/pb
        #

        #
        if grad:
            #
            dp=(2*b*X*np.sin(b*sxx)*pb+2*d*X*pa)/pb**2

        return (p, dp) if grad else p


funDropWave = FunDropWave()
