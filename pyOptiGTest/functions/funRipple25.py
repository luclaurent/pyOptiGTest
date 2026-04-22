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


class FunRipple25(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 14/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: x=0 >> f(-0.1,-0.1)=-2
        Design space 0<xi<1
        """

        #constants
        a=2
        b=0.1
        c=0.8
        d=5*np.pi

        #evaluation and derivatives
        pa=(X-b)/c
        pb=-a*np.log(a)*pa**2
        pc=-np.exp(pb)
        sd=np.sin(d*X)
        pd=sd**6
        cal=pc*pd
        #
        p=np.sum(cal, axis=1)

        if grad:
            dpd=6*d*sd**5*np.cos(d*X)
            dpb=-2*a*np.log(a)*pa*pc/c
            #
            dp=dpb*pd+pc*dpd


        return (p, dp) if grad else p


funRipple25 = FunRipple25()
