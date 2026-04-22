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


class FunSchaffer4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schaffer 4 function
        L. LAURENT -- 19/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,1.253115)=0.2924385
        Design space -100<xi<100
        """

        #constants
        a=0.5
        b=1
        c=0.001

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        xay=x**2+y**2
        xmy=x**2-y**2
        sxy=np.sin(xmy)
        csxy=np.cos(sxy)
        pa=csxy**2-a
        pb=b+c*xay**2
        #
        p=a+pa/pb

        if grad:
            #
            dpT=-2*X*np.cos(xmy)[:, None]*2*np.sin(sxy)[:, None]*csxy[:, None]
            dpax=dpT[:, 0]
            dpay=-dpT[:, 1]
            dpbx=4*c*x*xay
            dpby=4*c*y*xay
            dp = np.zeros_like(X)
            dp[:, 0]=(dpax*pb-pa*dpbx)/pb**2
            dp[:, 1]=(dpay*pb-pa*dpby)/pb**2

        return (p, dp) if grad else p


funSchaffer4 = FunSchaffer4()
