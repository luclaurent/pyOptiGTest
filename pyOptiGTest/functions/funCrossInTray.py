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


class FunCrossInTray(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 05/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x1,x2,x3,x4)=-2.06261218 for
        {(1.349406685353340,1.349406608602084),
        (-1.349406685353340,1.349406608602084),
        (1.349406685353340,-1.349406608602084),
        (-1.349406685353340,-1.349406608602084)}
        Design space: -10<xi<10
        """

        #constants
        a=1e-4
        b=100
        c=np.pi
        d=1
        e=0.1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        sx=np.sin(xxx)
        sy=np.sin(yyy)
        gxy=np.sqrt(xxx**2+yyy**2)
        hxy=np.exp(np.abs(b-gxy/c))
        kxy=np.abs(sx*sy*hxy)
        p=-a*(kxy+d)**e
        #
        if grad:
            cx=np.cos(xxx)
            cy=np.cos(yyy)
            #
            dgx=xxx/gxy
            dgy=yyy/gxy
            #
            dhx=hxy*np.sign(b-gxy/c)*(-1/c*dgx)
            dhy=hxy*np.sign(b-gxy/c)*(-1/c*dgy)
            #
            dkx=(cx*sy*hxy+sx*sy*dhx)*np.sign(sx*sy*hxy)
            dky=(cy*sx*hxy+sx*sy*dhy)*np.sign(sx*sy*hxy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=-a*e*dkx*(kxy+d)**(e-1)
            dp[:, 1]=-a*e*dky*(kxy+d)**(e-1)

        return (p, dp) if grad else p


funCrossInTray = FunCrossInTray()
