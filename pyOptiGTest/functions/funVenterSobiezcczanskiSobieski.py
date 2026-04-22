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


class FunVenterSobiezcczanskiSobieski(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Venter-Sobiezcczanski-Sobieski function
        L. LAURENT -- 27/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,0)=-400
        Design space -50<xi<50
        """

        #constants
        a=100
        c=30

        #evaluation and derivatives
        #
        cx=np.cos(X)
        ccx=np.cos(X**2/c)
        #
        pA=X**2-a*cx**2-a*ccx
        #
        p=np.sum(pA, axis=1)

        if grad:
            #
            sx=np.sin(X)
            scx=np.sin(X**2/c)
            #
            dp=2*X+2*a*cx*sx+2*a/c*X*scx

        return (p, dp) if grad else p


funVenterSobiezcczanskiSobieski = FunVenterSobiezcczanskiSobieski()
