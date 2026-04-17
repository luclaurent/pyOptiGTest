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


class FunXinSheYang2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Xin-She-Yang 2 function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -2pi<xi<2pi
        """

        #evaluation and derivatives
        pa=np.abs(X)
        pb=np.sin(X**2)
        #
        pA=np.sum(pa, axis=1)
        pB=np.sum(pb, axis=1)
        pC=np.exp(pB)
        #
        p=pA/pC

        if grad:
            dp=(np.sign(X)*pC-2*X*np.cos(X**2)*pC*pA)/pC**2

        return (p, dp) if grad else p


funXinSheYang2 = FunXinSheYang2()
