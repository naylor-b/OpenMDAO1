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

        root.add('p1', IndepVarComp('xC', 0.0), promotes=['*'])
        root.add('p2', IndepVarComp('xI', 0), promotes=['*'])
        root.add('comp', BranninInteger(), promotes=['*'])

        prob.driver = Branch_and_Bound()

        prob.driver.add_desvar('xI', lower=-5, upper=10)
        prob.driver.add_objective('f')

        prob.driver.sampling = {'xI' : np.linspace(0.0, 1.0, num=25)}

        prob.setup(check=False)
        prob.run()
        
        # Find an integer solution close to the floating-point mininum at (pi, 2.275)
        prob['xC'] = 2.275

        # Optimal solution
        assert_rel_error(self, prob['xI'], 3, 1e-5)
        #assert_rel_error(self, prob['f'], 2.38801229, 1e-5)

if __name__ == "__main__":
    unittest.main()
