""" Test for the Driver class -- basic driver interface."""

import unittest

import numpy as np

from openmdao.core.driver import Driver
from openmdao.core.options import OptionsDictionary
from openmdao.core.problem import Problem
from openmdao.test.sellar import SellarDerivatives

class MySimpleDriver(Driver):

    def __init__(self):
        super(MySimpleDriver, self).__init__()

        # What we support
        self.supports['Inequality Constraints'] = True
        self.supports['Equality Constraints'] = False
        self.supports['Linear Constraints'] = False
        self.supports['Multiple Objectives'] = False

        # My driver options
        self.options = OptionsDictionary()
        self.options.add_option('tol', 1e-4)
        self.options.add_option('maxiter', 10)

        self.alpha = .01
        self.violated = []

    def run(self, problem):
        """ Mimic a very simplistic unconstrained optimization."""

        # Get dicts with pointers to our vectors
        params = self.get_parameters()
        objective = self.get_objectives()
        constraints = self.get_constraints()

        param_list = params.keys()
        objective_names = objective.keys()
        constraint_names = constraints.keys()
        unknown_list = objective_names + constraint_names

        itercount = 0
        while itercount < self.options['maxiter']:

            # Run the model
            problem.root.solve_nonlinear()

            # Calculate gradient
            J = problem.calc_gradient(param_list, unknown_list)

            for key1 in objective_names:
                for key2 in param_list:

                    grad = J[key1][key2].dot(objective[key1])
                    new_val = params[key2] - self.alpha*grad

                    # Set parameter
                    self.set_param(key2, new_val)

            self.violated = []
            for name, val in constraints.items():
                if np.norm(val) > 0.0:
                    self.violated.append(name)



class TestDriver(unittest.TestCase):

    def test_mydriver(self):

        top = Problem()
        root = top.root = SellarDerivatives()

        top.driver = MySimpleDriver()
        top.driver.add_param('pz.z', low=-100.0, high=100.0)

        top.driver.add_objective('need to add expression comp')
        top.driver.add_constraint('need to add expression comp')

if __name__ == "__main__":
    unittest.main()
