def ogdStep(x, Gf, **parms):
    """
    ogdStep(t, x, (f, Gf), Proj) computes the update step for a simple online
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
