""" Testing the Branch and Bround driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp
from openmdao.drivers.branch_and_bound import Branch_and_Bound
from openmdao.test.branin import BranninInteger
from openmdao.test.util import assert_rel_error


class TestBranchAndBounddriver(unittest.TestCase):

    def test_brannin_just_opt_integer(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BranninInteger(), promotes=['*'])

        prob.driver = Branch_and_Bound()
        prob.driver.options['use_surrogate'] = True
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_objective('f')

        npt = 15
        prob.driver.sampling = {'xI' : np.linspace(0.0, 1.0, num=npt).reshape(npt, 1)}

        prob.setup(check=False)

        # Find an integer solution at a point where we know the minimum is in
        # the middle
        prob['xC'] = 13.0

        prob.run()

        # Optimal solution
        assert_rel_error(self, prob['xI'], -3, 1e-5)
        assert_rel_error(self, prob['f'], 1.62329296, 1e-5)

if __name__ == "__main__":
    unittest.main()
