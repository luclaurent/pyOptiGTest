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


class FunSchmidtVetters(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schmidt-Vetters's function
        L. LAURENT -- 19/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0.78547,0.78547,0.78547)=2.99845
        Design space 0<xi<10 (other minima outside of this space)
        """

        #constants
        a=1
        b=np.pi
        c=2

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        z=X[:, 2]
        #
        pa=a+(x-y)**2
        ga=a/pa
        pb=(b*y+z)/c
        gb=np.sin(pb)
        pc=((x+y)/y-c)
        gc=np.exp(pc**2)
        #
        p=ga+gb+gc

        if grad:
            #
            ax=-2*a*(x-y)/pa**2
            ay=-ax
            by=b/c*np.cos(pb)
            bz=1/c*np.cos(pb)
            cx=2/y*pc*gc
            cy=-2*x/y**2*pc*gc
            #
            dp=np.zeros_like(X)
            dp[:, 0]=ax+cx
            dp[:, 1]=ay+by+cy
            dp[:, 2]=bz

        return (p, dp) if grad else p


funSchmidtVetters = FunSchmidtVetters()
