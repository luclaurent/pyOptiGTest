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


class FunDeb2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Deb2's function
        L. LAURENT -- 06/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        5^np global minimum :
        design space -1<xi<1
        """

        #constans
        a=6
        b=5*np.pi
        c=2*np.log(2)
        d=0.1
        e=0.8

        #evaluation and derivatives
        nS=X.shape[1]
        sx=np.sin(b*X)
        ex=np.exp(-c*((X-d)/e)**2)
        p=-1/nS*np.sum(ex*sx**a, axis=1)
        if grad:
            cx=np.cos(b*X)
            dex=-2*c/e*(X-d)/e
            #
            dp=-1/nS*ex*(b*a*cx*sx**(a-1)+sx**a*dex)

        return (p, dp) if grad else p


funDeb2 = FunDeb2()
