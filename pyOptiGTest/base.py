"""
    pyOptiGTest - Base abstract class for test functions.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

from abc import ABC, abstractmethod


class TestFunction(ABC):
    """Abstract base class for all optimization test functions.

    Subclasses must implement the ``evaluate`` method.
    Instances are callable: ``f(X)`` is equivalent to ``f.evaluate(X)``.
    """

    @abstractmethod
    def evaluate(self, X, grad=False):
        """Evaluate the function at *X*.

        Parameters
        ----------
        X : numpy.ndarray
            Input array of shape ``(n_samples, n_vars)``.
        grad : bool, optional
            If ``True``, also return the gradient.

        Returns
        -------
        p : numpy.ndarray
            Function values, shape ``(n_samples,)``.
        dp : numpy.ndarray, optional
            Gradient, shape ``(n_samples, n_vars)``; only when *grad* is ``True``.
        """

    def __call__(self, X, grad=False):
        return self.evaluate(X, grad)
