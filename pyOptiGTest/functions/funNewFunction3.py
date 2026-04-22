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


class FunNewFunction3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        New Function 3 function
        L. LAURENT -- 21/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-1.019829 for x=[-1.9862 -10]
        Design space: -10<xi<10
        """
        #constants
        a=0.01
        b=0.1

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        cx=np.cos(x)
        cy=np.cos(y)
        sx=np.sin(x)
        sy=np.sin(y)
        scxy=np.sin((cx+cy)**2)
        csxy=np.cos((sx+sy)**2)
        #
        pa=(x+scxy**2+csxy**2)
        p=a*x+b*y+pa**2
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            ccxy=np.cos((cx+cy)**2)
            ssxy=np.sin((sx+sy)**2)
            #
            dp[:, 0]=a+2*pa*(1-4*sx*(cx+cy)*scxy*ccxy-4*cx*(sx+sy)*ssxy*csxy)
            dp[:, 1]=b+2*pa*(-4*sy*(cx+cy)*ccxy*scxy-4*cy*(sx+sy)*ssxy*csxy)
            #

        return (p, dp) if grad else p


funNewFunction3 = FunNewFunction3()
