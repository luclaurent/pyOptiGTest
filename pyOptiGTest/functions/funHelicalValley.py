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


class FunHelicalValley(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Helical Valley function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2,x3)=0 for (x1,x2,x3)=(1,0,0)
        Design space: -10<xi<10
        """
        #constants
        a=100
        b=10
        c=1
        d=1/(2*np.pi)

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]
        zzz=X[:, 2]

        #evaluation and derivatives
        th=np.arctan2(yyy,xxx)*d
        pa=yyy-b*th
        pb=np.sqrt(xxx**2+yyy**2)-c
        #
        p=a*(pa**2+pb**2)+zzz**2

        if grad:
            dthX=-d*yyy/(xxx**2+yyy**2)
            dthY=d*xxx/(xxx**2+yyy**2)
            pbX=2*a*xxx/np.sqrt(xxx**2+yyy**2)*pb
            pbY=2*a*yyy/np.sqrt(xxx**2+yyy**2)*pb
            paX=-2*a*b*dthX*pa
            paY=2*a*(1-b*dthY)*pa
            #
            dp = np.zeros_like(X)
            dp[:, 0]=paX+pbX
            dp[:, 1]=paY+pbY
            dp[:, 2]=2*zzz

        return (p, dp) if grad else p


funHelicalValley = FunHelicalValley()
