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


class FunBukin06(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 6's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=0.01
        c=10

        #evaluation and derivatives
        pa=X[:, 1]-b*X[:, 0]**2
        spa=np.sqrt(np.abs(pa))
        pb=X[:, 0]+c
        p=a*spa+b*np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-a*b*np.sign(pa)*X[:, 0]/spa+b*np.sign(pb)
            dp[:, 1]=a*np.sign(pa)/(2*spa)

        return (p, dp) if grad else p


funBukin06 = FunBukin06()
