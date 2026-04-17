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


class FunBukin18(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 18's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=15
        c=80
        d=21
        e=100
        f=17
        g=90


        #evaluation and derivatives
        pa=X[:, 1]+b*X[:, 0]+c
        pb=X[:, 1]-d*X[:, 0]-e
        pc=e*X[:, 0]+X[:, 1]-e
        pd=f*X[:, 0]+X[:, 1]+g
        p=a*np.abs(pa*pb*pc)+np.abs(pd)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=a*(b*pb*pc-d*pa*pc+e*pa*pb)*np.sign(pa*pb*pc)+np.sign(pd)
            dp[:, 1]=a*(pb*pc+pa*pc+pa*pb)*np.sign(pa*pb*pc)+np.sign(pd)

        return (p, dp) if grad else p


funBukin18 = FunBukin18()
