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


class FunGiunta(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Giunta function
        L. LAURENT -- 16/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0.06447042053690566 for (0.4673200277395354, 0.4673200169591304)
        Design space: -1<xi<1
        """

        #constants
        a=0.6
        b=16/15
        c=1
        d=1/50
        e=4

        #evaluation and derivatives
        pa=np.sin(b*X-c)
        pb=np.sin(e*(b*X-c))
        pc=pa+pa**2+d*pb
        p=a+np.sum(pc, axis=1)
        #
        if grad:
            dp=b*np.cos(b*X-c)+2*b*pa*np.cos(b*X-c)+d*e*b*np.cos(e*(b*X-c))

        return (p, dp) if grad else p


funGiunta = FunGiunta()
