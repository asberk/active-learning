import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt


def main():
    nMax = 100
    TMax = 10000
    result = SimpleMultiArmedBandit(n=nMax, T=TMax, delta=.5,
                                    algOCO=OnlineGradDescStep,
                                    actionLoss=inverseLoss,
                                    actionLossVerbose=True)
    for j in range(TMax-1000, TMax):
        plt.plot(result[:, j], color=plt.cm.Blues((j+1000-TMax)/1001))
    plt.show()


def SimpleMultiArmedBandit(n, T, delta, algOCO, actionLoss, **parms):
    """
    Simple multi-armed bandit algorithm
    """
    init = parms.get('init', {})
    ogdProj = parms.get('ogdProj', ProjSimplex)
    ogdStep = parms.get('ogdStep', .1)
    alVerbose = parms.get('actionLossVerbose', False)

    fHat = np.zeros(T)
    fHat[0] = init.get('fHat0', 0)
    x = np.zeros((n, T+1))
    x[:, 0] = init.get('x0', np.ones(n) / n)
    ellHat = np.zeros((n, T))
    ellHat[:, 0] = init.get('ellHat0', np.zeros(n))

    selectedActionNumber = np.zeros(T, dtype=np.int)
    observedLoss = np.zeros(T)

    for t in range(T):
        if (np.random.rand(1) > delta):
            selectedActionNumber[t] = np.random.randint(n)
            # print(x[:, t])
            observedLoss[t] = actionLoss(selectedActionNumber[t], n,
                                         t=t, T=T, x=x[:, t],
                                         verbose=alVerbose)
            ellHat[selectedActionNumber[t], t] = n * observedLoss[t] / delta
            fHat[t] = np.dot(ellHat[:, t], x[:, t])
            x[:, t+1] = algOCO(x[:, t], lambda x: ellHat[:, t],
                               Proj=ogdProj, eta=ogdStep)
        else:
            selectedActionNumber[t] = np.random.choice(np.arange(n, dtype=np.int), p=x[:, t])
            x[:, t+1] = algOCO(x[:, t], lambda x: ellHat[:, t],
                               Proj=ogdProj, eta=ogdStep)
    # return optimal sampling strategy after T iterations
    return x


def OnlineGradDescStep(x, Gf, **parms):
    """
    OnlineGD(t, x, (f, Gf), Proj) computes the update step for a simple online
    gradient descent algorithm.

    Input:
       t : the current time-step
       x : the current pmf on actions?
      Gf : the gradient, Gf, of the current loss function.
           Note that in this implementation, Gf is a function.
    parms: keyword arguments specifying parameters for the online GD algorithm
      |- Proj : function computing the projection onto the feasible set
      |-  eta : step size to use
    Output:
    xnew : the new distribution from which our algorithm samples actions.

    """
    Proj = parms.get('Proj', lambda x: x)
    eta = parms.get('eta', .1)
    xnew = x - eta * Gf(x)
    xnew = Proj(xnew)
    return xnew


def ProjSimplex(y):
    """
    Computes the projection onto the simplex using
    Algorithm 1 of (Chen & Ye, 2011).
    """
    n = y.size
    J = np.argsort(y)
    i = n-1
    while True:
        ti = (np.sum(y[J][i:])-1)/(n-i)
        i -= 1
        if ti >= y[J][i]:
            tHat = ti
            break
        elif i == 0:
            tHat = (np.sum(y)-1)/n
            break
    return np.maximum(0, y - tHat)


def userDefinedLoss(actionNumber, n, **kwargs):
    verbose = kwargs.get('verbose', False)
    if verbose:
        t = kwargs.get('t', None)
        T = kwargs.get('T', None)
        x = kwargs.get('x', None)
        if (t is not None) and (T is not None):
            print('Percent complete: {}%'.format(np.round(100*(t+1)/T, 2)))
        if x is not None:
            print('Machine\'s pmf over actions is:')
            print(x)
    print('Machine selected action {} of {}.'.format(actionNumber, n))
    print('How would you like to penalize this action?')
    userDefinedLoss = input('(Enter a value between 0 and 1): ')
    userDefinedLoss = float(Fraction(userDefinedLoss))
    userDefinedLoss = np.maximum(0, np.minimum(1, userDefinedLoss))
    return userDefinedLoss


def inverseLoss(actionNumber, n, **kwargs):
    verbose = kwargs.get('verbose', False)
    if verbose:
        t = kwargs.get('t', None)
        T = kwargs.get('T', None)
        x = kwargs.get('x', None)
        if (t is not None) and (T is not None):
            print('Percent complete: {}%'.format(np.round(100*(t+1)/T, 2)))
        if x is not None:
            print('Machine\'s pmf over actions is:')
            print(x)
    inverseLoss = np.maximum(0, np.minimum(1, actionNumber/n))
    return inverseLoss


if __name__ == "__main__":
    print('Running interactive simple active learning' +
          'example from  command line...')
    main()
