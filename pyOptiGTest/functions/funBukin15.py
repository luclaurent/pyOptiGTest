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


class FunBukin15(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 15's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=10
        c=25
        d=0.1
        e=75


        #evaluation and derivatives
        pa=X[:, 1]+X[:, 0]**2+b*X[:, 0]-c
        pb=X[:, 1]+b*X[:, 0]+e
        p=a*np.abs(pa)+d*np.abs(pb)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=a*(2*X[:, 0]+b)*np.sign(pa)+d*b*np.sign(pb)
            dp[:, 1]=a*np.sign(pa)+d*np.sign(pb)

        return (p, dp) if grad else p


funBukin15 = FunBukin15()
