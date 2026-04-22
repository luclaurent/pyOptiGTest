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


class FunVincent(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Vincent function
        L. LAURENT -- 27/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(7.70628098,...,7.70628098)=-n
        Design space 0.25<xi<10
        """

        #constants
        a=10
        #evaluation and derivatives
        #
        ll=np.log(X)
        sl=np.sin(a*ll)
        #
        p=-np.sum(sl, axis=1)

        if grad:
            #
            dp=-a/X*np.cos(a*ll)

        return (p, dp) if grad else p


funVincent = FunVincent()
