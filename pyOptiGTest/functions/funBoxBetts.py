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


class FunBoxBetts(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        optiGTest - set of testing functions    A toolbox to easy manipulate functions.
        Copyright (C) 2017  Luc LAURENT <luc.laurent@lecnam.net>
        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
        one local minimum
        1 global minimum : x=(1, 10, 1) >> f(x)=0
        design space 0.9<x1,x3<1.2 and 9<x2<11.2
        """


        #constants
        a=0.1
        nbt=10

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]
        zzz=X[:, 2]

        p=np.zeros_like(xxx)

        funG=lambda x1,x2,x3,j: np.exp(-a*j*x1)-np.exp(-a*j*x2)-(np.exp(-a*j)-np.exp(-j))*x3
        for it in range(1, nbt+1):
            p=p+funG(xxx,yyy,zzz,it)**2

        if grad:
            dp=np.zeros((xxx.shape[0], 3))
            for it in range(1, nbt+1):
                dp[:, 0]=dp[:, 0]-a*it*np.exp(-a*it*xxx)
                dp[:, 1]=dp[:, 1]+a*it*np.exp(-a*it*yyy)
                dp[:, 2]=dp[:, 2]-np.exp(-a*it)+np.exp(-it)

        return (p, dp) if grad else p


funBoxBetts = FunBoxBetts()
