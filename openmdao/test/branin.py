""" Test objects for the single-discipline Brannin (or Brannin-Hoo) problem.

The Branin, or Branin-Hoo, function has three global minima. The recommended
values of a, b, c, r, s and t are:
  a = 1
  b = 5.1/(4pi2),
  c = 5/pi,
  r = 6,
  s = 10 and
  t = 1/(8pi).

This function is usually evaluated on the square x0 ~ [-5, 10], x1 ~ [0, 15].

The global minimum can be found at f(x) = 0.397887
"""

import numpy as np

from openmdao.core.component import Component


class Brannin(Component):
    """ The Brannin test problem. This version contains a continuous and an
    integer parameter. """

    def __init__(self):
        super(Brannin, self).__init__()

        # Inputs
        self.add_param('x0', 0.0)
        self.add_param('x1', 0.0)

        # Outputs
        self.add_output('f', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        x0 = params['x0']
        x1 = params['x1']

        a = 1.0
        b = 5.1/(4.0*np.pi**2)
        c = 5.0/np.pi
        d = 6.0
        e = 10.0
        f = 1.0/(8.0*np.pi)

        unknowns['f'] = a*(x1 - b*x0**2 + c*x0 - d)**2 + e*(1-f)*np.cos(x0) + e

    def linearize(self, params, unknowns, resids):
        """ Provide the Jacobian"""
        J = {}
        x0 = params['x0']
        x1 = params['x1']

        a = 1.0
        b = 5.1/(4.0*np.pi**2)
        c = 5.0/np.pi
        d = 6.0
        e = 10.0
        f = 1.0/(8.0*np.pi)

        J['f', 'x1'] = 2.0*a*(x1 - b*x0**2 + c*x0 - d)
        J['f', 'x0'] = 2.0*a*(x1 - b*x0**2 + c*x0 - d)*(-2.*b*x0 + c) - e*(1.-f)*np.sin(x0)

        return J

class BranninInteger(Component):
    """ The Brannin test problem. This version contains a continuous and an
    integer parameter. """

    def __init__(self):
        super(BranninInteger, self).__init__()

        # Inputs
        self.add_param('xI', 0)
        self.add_param('xC', 0.0)

        # Outputs
        self.add_output('f', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Define the function f(xI, xC)
        Here xI is integer and xC is continuous"""

        x0 = params['xI']
        x1 = params['xC']

        a = 1.0
        b = 5.1/(4.0*np.pi**2)
        c = 5.0/np.pi
        d = 6.0
        e = 10.0
        f = 1.0/(8.0*np.pi)

        unknowns['f'] = a*(x1 - b*x0**2 + c*x0 - d)**2 + e*(1-f)*np.cos(x0) + e

    def linearize(self, params, unknowns, resids):
        """ Provide the Jacobian"""
        J = {}
        x0 = params['xI']
        x1 = params['xC']

        a = 1.0
        b = 5.1/(4.0*np.pi**2)
        c = 5.0/np.pi
        d = 6.0
        e = 10.0
        f = 1.0/(8.0*np.pi)

        J['f', 'xC'] = 2.0*a*(x1 - b*x0**2 + c*x0 - d)

        # Derivative of integer var, if you could differentiate it.
        # J['f', 'xI'] = 2.0*a*(x1 - b*x0**2 + c*x0 - d)*(-2.*b*x0 + c) - e*(1.-f)*np.sin(x0)

        return J