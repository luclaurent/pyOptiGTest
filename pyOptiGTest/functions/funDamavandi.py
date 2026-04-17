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


class FunDamavandi(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 05/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 global minimum : f(2,2)=0
        Design space: 0<xi<14
        """

        #constants
        a=1
        b=np.pi
        c=2
        d=7

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pax=xxx-c
        pay=yyy-c
        sx=np.sin(b*pax)
        sy=np.sin(b*pay)
        kxy=sx*sy/(b**2*pax*pay)
        #
        pbx=xxx-d
        pby=yyy-d
        hxy=c+pbx**2+c*pby**2
        #
        p=(a-np.abs(kxy)**5)*hxy
        #
        if grad:
            cx=np.cos(b*pax)
            cy=np.cos(b*pay)
            #
            dkx=b*cx*sy/(b**2*pax*pay)+b**2*pax*kxy/(b**2*pax*pay)
            dky=b*cy*sx/(b**2*pax*pay)+b**2*pbx*kxy/(b**2*pax*pay)
            #
            dhx=2*pbx
            dhy=2*c*pby
            #
            dp = np.zeros_like(X)
            dp[:, 0]=dhx*(a-np.abs(kxy)**5)-5*dkx*np.sign(kxy)*kxy**4*hxy
            dp[:, 1]=dhy*(a-np.abs(kxy)**5)-5*dky*np.sign(kxy)*kxy**4*hxy

        return (p, dp) if grad else p


funDamavandi = FunDamavandi()
