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


class FunMishra09(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 9 function
        global minimum : f(x)=0 for x=(1,2,3)
        Design space: -10<xi<10
        """
        d = [2, 5, 4, -2, -18]
        e = -22
        f = [8, 2, 2, 3, -52]

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]

        a = d[0]*xxx**3 + d[1]*xxx*yyy + d[2]*zzz + d[3]*xxx**2*zzz + d[4]
        b = xxx + yyy**3 + xxx*yyy**2 + xxx*zzz**2 + e
        c = f[0]*xxx**2 + f[1]*yyy*zzz + f[2]*yyy**2 + f[3]*yyy**3 + f[4]
        k = xxx + yyy - zzz

        pa = a*b**2*c + a*b*c**2 + b**2 + k**2
        p = pa**2

        if grad:
            daX = 3*d[0]*xxx**2 + d[1]*yyy + 2*d[3]*xxx*zzz
            daY = d[1]*xxx
            daZ = d[2] + d[3]*xxx**2
            dbX = 1 + zzz**2 + yyy**2
            dbY = 3*yyy**2 + 2*yyy*xxx
            dbZ = 2*xxx*zzz
            dcX = 2*f[0]*xxx
            dcY = f[1]*zzz + 2*f[2]*yyy + 3*f[3]*yyy**2
            dcZ = f[1]*yyy
            dkX = 1
            dkY = 1
            dkZ = -1

            dp = np.zeros_like(X)
            dp[:, 0] = 2*pa*(daX*b**2*c + 2*a*dbX*b*c + a*b**2*dcX + daX*b*c**2 + a*dbX*c**2 + 2*a*b*dcX*c + 2*dbX + 2*dkX*k)
            dp[:, 1] = 2*pa*(daY*b**2*c + 2*a*dbY*b*c + a*b**2*dcY + daY*b*c**2 + a*dbY*c**2 + 2*a*b*dcY*c + 2*dbY + 2*dkY*k)
            dp[:, 2] = 2*pa*(daZ*b**2*c + 2*a*dbZ*b*c + a*b**2*dcZ + daZ*b*c**2 + a*dbZ*c**2 + 2*a*b*dcZ*c + 2*dbZ + 2*dkZ*k)

        return (p, dp) if grad else p



funMishra09 = FunMishra09()
