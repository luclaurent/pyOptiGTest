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


class FunXinSheYang4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Xin-She-Yang 4 function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=-1
        Design space -10<xi<10
        """


        #evaluation and derivatives
        pa=np.sin(X)
        pAA=np.sum(pa**2, axis=1)
        #
        pb=X**2
        pB=np.sum(pb, axis=1)
        pBB=np.exp(-pB)
        #
        pc=np.sin(np.sqrt(np.abs(X)))
        pC=np.sum(pc**2, axis=1)
        pCC=np.exp(-pC)
        #
        p=(pAA-pBB)*pCC

        if grad:
            dp=2*(pa*np.cos(X)+X*pBB)*pCC -(pAA-pBB)*np.sign(X)/np.sqrt(np.abs(X))*pc*np.cos(np.sqrt(np.abs(X)))*pCC

        return (p, dp) if grad else p


funXinSheYang4 = FunXinSheYang4()
