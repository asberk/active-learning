import numpy as _np
from .Projection import ProjSimplex


def simpleMAB(n, T, delta, algOCO, actionLoss, **parms):
    """
    Simple multi-armed bandit algorithm
    Input:
             n : The number of actions from which the algorithm can
                 choose. Note that the algorithm will always choose a random
                 integer from {0, 1, ..., n-1}. It is thus up to the user to
                 define a loss that maps integers to outcomes, actions, losses
                 or consequences.
             T : the number of time steps for which to run the online
                 optimization.
         delta : The paramter controlling the simple multi-armed bandit. delta
                 is the probability that the Bernoulli random variable will be
                 equal to 1, and thus is the probability that the algorithm
                 "explores" by choosing an action uniformly at random, rather
                 than sampling from the empirical distribution it is
                 optimizing.
        algOCO : The online convex optimization algorithm used by the
                 MAB. Projection and Step-Size can be passed as ogdProj and
                 ogdStep, respectively.
    actionLoss : The loss/regret function used by the algorthm. For example,
                 the loss could be inversely proportional to the action number
                 selected by the algorithm (inverseLoss).
         parms
           |       init : Initial conditions for the algorithm.
           |    ogdProj : Projection method to be used by algOCO
                          (default: ProjSimplex; assumes that we are recovering
                          a decision strategy that is a random variable with a
                          known but possibly time-evolving law)
           |    ogdStep : The step-size to be used by the online gradient
                          descent algorithm.
           |  alVerbose : how much information to print out.
    """
    init = parms.get('init', {})
    ogdProj = parms.get('ogdProj', ProjSimplex)
    ogdStep = parms.get('ogdStep', .1)
    alVerbose = parms.get('actionLossVerbose', False)

    fHat = _np.zeros(T)
    fHat[0] = init.get('fHat0', 0)
    x = _np.zeros((n, T+1))
    x[:, 0] = init.get('x0', _np.ones(n) / n)
    ellHat = _np.zeros((n, T))
    ellHat[:, 0] = init.get('ellHat0', _np.zeros(n))

    selectedActionNumber = _np.zeros(T, dtype=_np.int)
    observedLoss = _np.zeros(T)

    for t in range(T):
        if (_np.random.rand(1) <= delta):
            selectedActionNumber[t] = _np.random.randint(n)
            # print(x[:, t])
            observedLoss[t] = actionLoss(selectedActionNumber[t], n,
                                         t=t, T=T, x=x[:, t],
                                         verbose=alVerbose)
            ellHat[selectedActionNumber[t], t] = n * observedLoss[t] / delta
            fHat[t] = _np.dot(ellHat[:, t], x[:, t])
            x[:, t+1] = algOCO(x[:, t], lambda x: ellHat[:, t],
                               Proj=ogdProj, eta=ogdStep)
        else:
            selectedActionNumber[t] = \
                _np.random.choice(_np.arange(n, dtype=_np.int),
                                 p=x[:, t])
            x[:, t+1] = algOCO(x[:, t], lambda x: ellHat[:, t],
                               Proj=ogdProj, eta=ogdStep)
    # return optimal sampling strategy after T iterations
    return x
