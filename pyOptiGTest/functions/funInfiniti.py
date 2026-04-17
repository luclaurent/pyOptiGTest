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


class FunInfiniti(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Infiniti function
        L. LAURENT -- 16/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=0
        Design space: -1<xi<1
        """
        #constants
        a=2

        #evaluation and derivatives
        pa=X**6
        pb=np.sin(1/X)+a
        #
        p=np.sum(pa*pb, axis=1)

        #
        if grad:
            #
            dp=6*X**5*pb-X**4*np.cos(1/X)

        return (p, dp) if grad else p


funInfiniti = FunInfiniti()
