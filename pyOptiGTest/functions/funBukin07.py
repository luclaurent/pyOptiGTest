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


class FunBukin07(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 7's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=25
        c=5
        d=np.exp(5)

        #evaluation and derivatives
        pa=b+X[:, 0]*X[:, 1]
        spa=np.sqrt(np.abs(pa))
        pb=X[:, 0]+np.exp(X[:, 1])-d+c
        spb=np.sqrt(np.abs(pb))
        p=a*spa+a*spb
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=a*np.sign(pa)*X[:, 1]/(2*spa)+a*np.sign(pb)/(2*spb)
            dp[:, 1]=a*np.sign(pa)*X[:, 0]/(2*spa)+a*np.sign(pb)*np.exp(X[:, 1])/(2*spb)

        return (p, dp) if grad else p


funBukin07 = FunBukin07()
