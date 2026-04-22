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


class FunStyblinskiTang(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Styblinski Tang function
        L. LAURENT -- 23/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-39.16616570377142*nbvar for xi=-1.903534018185960
        Design space: -5<xi<5
        """
        #constants
        a=16
        b=5

        #evaluation and derivatives
        pa=X**4-a*X**2+b*X
        #
        p=np.sum(pa*1/2, axis=1)
        #
        if grad:
            #
            dp=1/2*(4*X**3-2*a*X+b)

        return (p, dp) if grad else p


funStyblinskiTang = FunStyblinskiTang()
