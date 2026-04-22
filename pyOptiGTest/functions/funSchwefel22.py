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


class FunSchwefel22(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 22 function
        L. LAURENT -- 19/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(0,...,0)=0
        Design space -100<xi<100
        """

        dim=X.shape[1]

        #evaluation and derivatives
        pa=np.abs(X)
        #
        p=np.sum(pa, axis=1)+np.prod(pa, axis=1)


        if grad:
            #
            # product-without-i: pd[:, i] = prod(|X|) / |X[:, i]|
            prod_pa = np.prod(pa, axis=1)
            pd = prod_pa[:, None] / pa
            #
            dp=np.sign(X)*(1+pd)

        return (p, dp) if grad else p


funSchwefel22 = FunSchwefel22()
