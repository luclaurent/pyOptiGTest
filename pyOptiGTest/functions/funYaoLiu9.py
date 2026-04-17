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


class FunYaoLiu9(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Yao-Liu 9 function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -5.12<xi<5.12
        """

        #constants
        a=10
        b=2*np.pi

        #evaluation and derivatives
        pa=X**2-a*np.cos(b*X)+a
        #
        p=np.sum(pa, axis=1)

        if grad:
            dp=2*X+a*b*np.sin(b*X)

        return (p, dp) if grad else p


funYaoLiu9 = FunYaoLiu9()
