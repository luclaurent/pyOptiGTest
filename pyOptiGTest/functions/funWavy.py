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


class FunWavy(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Wavy function
        L. LAURENT -- 27/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -pi<xi<pi
        """

        #constants
        a=1
        b=1/2
        k=10
        nbvar=X.shape[1]

        #evaluation and derivatives
        cx=np.cos(k*X)
        ex=np.exp(-b*X**2)
        pcx=cx*ex
        #
        p=a-1/nbvar*np.sum(pcx, axis=1)

        if grad:
            #
            sx=np.sin(k*X)
            #
            dp=1/nbvar*ex*(k*sx+X*cx)


        return (p, dp) if grad else p


funWavy = FunWavy()
