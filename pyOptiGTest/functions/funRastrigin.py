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


class FunRastrigin(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Rastrigin function
        L. LAURENT -- 21/02/2012 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        numerous local minima
        1 minimum global: x=(0,0,...,0) >> f(x)=0
        Design space -5.12<xi<5.12
        [TZ89] A. T\�orn and A. Zilinskas. "Global Optimization". Lecture Notes in Computer Science, No 350, Springer-Verlag, Berlin,1989.
        [MSB91] H. M\�uhlenbein, D. Schomisch and J. Born. "The Parallel Genetic Algorithm as Function Optimizer ". Parallel Computing, 17, pages 619-632,1991.
        """

        #constants
        coef=10
        nbvar=X.shape[1]

        #evaluation and derivatives
        cal=X**2-coef*np.cos(2*np.pi*X)
        p=coef*nbvar+np.sum(cal, axis=1)

        if grad:
            dp=2*X+2*coef*np.pi*np.sin(2*np.pi*X)

        return (p, dp) if grad else p


funRastrigin = FunRastrigin()
