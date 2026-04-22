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


class FunUrsemWaves(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ursem Waves function
        L. LAURENT -- 24/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(1.2,1.2)=-8.5536
        Design space -2<xi<2
        """

        #constants
        a=0.9
        b=4.5
        c=4.7
        d=2
        e=2.5*np.pi

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        xyA=y**2-b*y**2
        xyB=d*x-y**2*(d+x)
        cdxy=c*np.cos(xyB)
        se=np.sin(e*x)
        #
        p=-a*x**2 +xyA*x*y +cdxy*se

        if grad:
            #
            dp=np.zeros_like(X)
            #
            sdxy=np.sin(xyB)
            #
            dp[:, 0]=-2*a*x+xyA*y -c*(d-y**2)*sdxy*se +e*cdxy*np.cos(e*x)
            dp[:, 1]=(1-b)*3*x*y**2 +2*c*y*(d+x)*sdxy*se

        return (p, dp) if grad else p


funUrsemWaves = FunUrsemWaves()
