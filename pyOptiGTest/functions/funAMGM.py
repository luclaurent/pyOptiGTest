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


class FunAMGM(TestFunction):
    def evaluate(self, X, grad=False):
        """
        AMGM function (Arthmetic Mean - Geometric Mean)
        L. LAURENT -- 13/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x1=x2=x3=...=xn
        Design space: 0<xi<10
        """

        #variables
        dim=X.shape[1]

        #evaluation and derivatives
        pa=np.sum(X, axis=1)
        pb=np.prod(X, axis=1)
        pc=1/dim*pa-pb**(1/dim)
        #
        p=pc**2

        if grad:
            #
            # dpc/dxi = 1/dim - (1/dim) * pb^(1/dim - 1) * pb / xi
            #         = 1/dim - pb^(1/dim) / (dim * xi)
            pdB = 1/dim - pb[:, None]**(1/dim) / (dim * X)
            #
            dp = 2 * pdB * pc[:, None]

        return (p, dp) if grad else p


funAMGM = FunAMGM()
