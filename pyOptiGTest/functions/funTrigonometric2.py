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


class FunTrigonometric2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Trigonometric 2 function
        L. LAURENT -- 23/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0.9,...,0.9)=1
        Design space -500<xi<500 (focus on 0<xi<2)
        """

        #constants
        a=1
        b=8
        c=7
        d=0.9
        e=6
        f=14

        #evaluation and derivatives
        xd=X-d
        xd2=xd**2
        scx=np.sin(c*xd2)
        sfx=np.sin(f*xd2)
        #
        pa=b*scx**2#+e*sfx**2+xd2
        #
        p=a+np.sum(pa, axis=1)

        if grad:
            #
            ccx=np.cos(c*xd2)
            cfx=np.cos(f*xd2)
            #
            dp=4*b*c*xd*ccx*scx +4*e*f*xd*cfx*sfx +2*xd

        return (p, dp) if grad else p


funTrigonometric2 = FunTrigonometric2()
