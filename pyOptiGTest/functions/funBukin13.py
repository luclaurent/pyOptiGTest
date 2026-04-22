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


class FunBukin13(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 13's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=100
        b=3
        c=0.5

        #evaluation and derivatives
        pa=X[:, 0]+b
        pb=X[:, 1]-c
        rho=np.sqrt(pa**2+pb**2)
        m=np.sin(b*rho)*pa-np.cos(b*rho)*pb
        p=rho+a/rho**2*m**2
        if grad:
            dRhoX=pa/rho
            dRhoY=pb/rho
            dp = np.zeros_like(X)
            dp[:, 0]=dRhoX-2*a/rho**3*dRhoX*(m**2) +2*a/rho**2*(b*dRhoX*np.cos(b*rho)*pa+np.sin(b*rho)+b*dRhoX*np.sin(b*rho)*pb)*m
            dp[:, 1]=dRhoY-2*a/rho**3*dRhoY*(m**2) +2*a/rho**2*(b*dRhoY*np.cos(b*rho)*pa-np.cos(b*rho)+b*dRhoY*np.sin(b*rho)*pb)*m

        return (p, dp) if grad else p


funBukin13 = FunBukin13()
