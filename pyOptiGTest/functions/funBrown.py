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


class FunBrown(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/05/2012 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        design space -1<xi<4
        """

        #handle functions
        funG=lambda x,y: np.exp((y**2+1)*np.log(x**2))+np.exp((x**2+1)*np.log(y**2))
        funDxG=lambda x,y: 2*(y**2+1)/x*np.exp((y**2+1)*np.log(x**2))+2*x*np.log(y**2)*np.exp((x**2+1)*np.log(y**2))
        funDyG=lambda x,y: 2*y*np.log(x**2)*np.exp((y**2+1)*np.log(x**2))+2*(x**2+1)/y*np.exp((x**2+1)*np.log(y**2))

        #evaluation and derivatives
        tSum=funG(X[:, :-1],X[:, 1:])
        p=np.sum(tSum, axis=1)
        if grad:
            sX=X.shape
            nbvar=sX[1]
            dp=np.zeros(sX)
            for ii in range(1, nbvar+1):
                if ii==1:
                    dp[:, ii-1]=funDxG(X[:, 0],X[:, 1])
                elif ii==nbvar:
                    dp[:, ii-1]=funDyG(X[:, nbvar-1-1],X[:, nbvar-1])
                else:
                    dp[:, ii-1]=funDxG(X[:, ii-1],X[:, ii+1-1]) +funDyG(X[:, ii-1-1],X[:, ii-1])

        return (p, dp) if grad else p


funBrown = FunBrown()
