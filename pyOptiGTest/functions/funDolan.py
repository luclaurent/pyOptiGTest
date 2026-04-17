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


class FunDolan(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 15/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x1,x2,x3,x4)=0
        Design space: -100<xi<100
        """

        #constants
        a=1.7
        b=01.5
        c=0.1
        d=0.2
        e=1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]
        zzz=X[:, 2]
        vvv=X[:, 3]
        www=X[:, 4]

        #evaluation and derivatives
        pa=xxx+a*yyy
        pb=np.sin(xxx)
        pc=b*zzz
        pd=c*vvv
        pe=np.cos(vvv+www-xxx)
        pf=d*www**2-yyy-e
        #
        pt=pa*pb-pc-pd*pe+pf
        p=np.abs(pt)
        #
        if grad:
            sf=np.sign(pt)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=sf*(np.cos(xxx)*pa+pb-pd*np.sin(vvv+www-xxx))
            dp[:, 1]=sf*(a*pb-1)
            dp[:, 2]=-sf*b
            dp[:, 3]=sf*(-c*pe+pd*np.sin(vvv+www-xxx))
            dp[:, 4]=sf*(pd*np.sin(vvv+www-xxx)+2*d*www)

        return (p, dp) if grad else p


funDolan = FunDolan()
