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


class FunBukin12(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 12's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=1000
        b=5

        #evaluation and derivatives
        pa=X[:, 0]+b
        pb=X[:, 1]+b
        rho=np.sqrt(pa**2+pb**2)
        pc=X[:, 0]+b-rho*np.cos(rho)
        pd=X[:, 1]+b+rho*np.sin(rho)
        p=a*np.abs(pc)+a*np.abs(pd)+rho
        if grad:
            dRhoX=pa/rho
            dRhoY=pb/rho
            dp = np.zeros_like(X)
            dp[:, 0]=a*np.sign(pc)*(1+rho*dRhoX*np.sin(rho)-dRhoX*np.cos(rho)) +a*np.sign(pd)*(rho*dRhoX*np.cos(rho)+dRhoX*np.sin(rho))+dRhoX
            dp[:, 1]=a*np.sign(pc)*(rho*dRhoY*np.sin(rho)-dRhoY*np.cos(rho)) +a*np.sign(pd)*(1+rho*dRhoY*np.cos(rho)+dRhoY*np.sin(rho))+dRhoY

        return (p, dp) if grad else p


funBukin12 = FunBukin12()
