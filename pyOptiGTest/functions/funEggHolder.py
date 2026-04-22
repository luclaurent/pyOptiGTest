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


class FunEggHolder(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(512.0, 404.2319) >> f(x)=-959.640662711
        design space -512<xi<512
        """

        #constants
        a=47
        b=0.5

        #variables
        dim=X.shape[1]

        #evaluations and derivatives
        pa=X[:, 1:]+a
        pb=X[:, 1:]+X[:, :-1]*0.5+a
        pc=X[:, :-1]-X[:, 1:]-a
        #
        sqpb=np.sqrt(np.abs(pb))
        sqpc=np.sqrt(np.abs(pc))
        sipb=np.sin(sqpb)
        sipc=np.sin(sqpc)
        #
        g=-pa*sipb-X[:, :-1]*sipc
        p=np.sum(g, axis=1)

        if grad:
            dgi=-b*pa*np.sign(pb)*1/(2*sqpb)*np.cos(sqpb) -sipc -X[:, :-1]*np.sign(pc)*1/(2*sqpc)*np.cos(sqpc)
            dgii=-sipb -pa*np.sign(pb)*1/(2*sqpb)*np.cos(sqpb) +X[:, :-1]*np.sign(pc)*1/(2*sqpc)*np.cos(sqpc)
            #
            dp=np.zeros_like(X)
            dp[:, 0]=dgi[:, 0]
            for it in range(2, dim-1+1):
                dp[:, it-1]=dgi[:, it-1]+dgii[:, it-1-1]
            dp[:, dim-1]=dgii[:, -1]

        return (p, dp) if grad else p


funEggHolder = FunEggHolder()
