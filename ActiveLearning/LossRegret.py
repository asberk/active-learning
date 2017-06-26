import numpy as _np
from fractions import Fraction as _Fraction


def userDefinedLoss(actionNumber, n, **kwargs):
    verbose = kwargs.get('verbose', False)
    if verbose:
        t = kwargs.get('t', None)
        T = kwargs.get('T', None)
        x = kwargs.get('x', None)
        if (t is not None) and (T is not None):
            print('Percent complete: {}%'.format(_np.round(100*(t+1)/T, 2)))
        if x is not None:
            print('Machine\'s pmf over actions is:')
            print(x)
    print('Machine selected action {} of {}.'.format(actionNumber, n))
    print('How would you like to penalize this action?')
    userDefinedLoss = input('(Enter a value between 0 and 1): ')
    userDefinedLoss = float(_Fraction(userDefinedLoss))
    userDefinedLoss = _np.maximum(0, _np.minimum(1, userDefinedLoss))
    return userDefinedLoss


def inverseLoss(actionNumber, n, **kwargs):
    verbose = kwargs.get('verbose', False)
    if verbose:
        t = kwargs.get('t', None)
        T = kwargs.get('T', None)
        x = kwargs.get('x', None)
        if (t is not None) and (T is not None):
            print('Percent complete: {}%'.format(_np.round(100*(t+1)/T, 2)))
        if x is not None:
            print('Machine\'s pmf over actions is:')
            print(x)
    inverseLoss = _np.maximum(0, _np.minimum(1, actionNumber/n))
    return inverseLoss
